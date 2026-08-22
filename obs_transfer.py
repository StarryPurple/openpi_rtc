#!/usr/bin/env python3
"""OBS transfer helper for the openpi_rtc bundles (industrial PC channel).

Only depends on the official Huawei Cloud OBS Python SDK (esdk-obs-python).
Install once:

    pip install -r requirements-obs.txt
    # or: uv pip install -r requirements-obs.txt

Usage (ref 是 object key，可带 openpi05/ 前缀；默认桶 openpi-rtc，可用
--bucket 换桶；endpoint 默认北京四，可用 --endpoint 或 OBS_ENDPOINT 换):
    python obs_transfer.py ls [prefix] [--bucket openpi-rtc] [--endpoint https://obs.cn-north-4.myhuaweicloud.com]
    python obs_transfer.py upload <local_file> <ref> [--part-mb 64] [--workers 4] [--force]
    python obs_transfer.py download <ref> <local_file>
    python obs_transfer.py rm <ref> --yes

Examples (defaults: bucket openpi-rtc, endpoint cn-north-4):
    # list everything under openpi05/
    python obs_transfer.py ls openpi05/

    # upload the inference bundle (工控机用)
    python obs_transfer.py upload /tmp/openpi_inference_bundle.tar.gz openpi05/inference_bundle.tar.gz

    # pull it on the industrial PC
    python obs_transfer.py download openpi05/inference_bundle.tar.gz /tmp/inference_bundle.tar.gz

Credentials:
    Fill OBS_ACCESS_KEY_ID / OBS_SECRET_ACCESS_KEY below, or set the
    OBS_AK / OBS_SK environment variables. The script never prints them.
    Bucket / endpoint defaults can be overridden with OBS_BUCKET /
    OBS_ENDPOINT environment variables.

Notes:
    - Files <=64MB use a single PUT (md5 user metadata attached and verified);
      larger files use the SDK's resumable multipart upload
      (checkpoint file "<local>.obs-upload.cp"; delete it, or pass --force,
      to restart). Multipart uploads also attach the md5 metadata, so the
      post-upload verification covers size and md5.
    - Downloads >512MB use the SDK's resumable downloadFile.
    - Alternative on machines that prefer the Huawei obsutil CLI (single
      binary, already configured on the dev machine):
        obsutil cp /tmp/inference_bundle.tar.gz obs://openpi-rtc/openpi05/inference_bundle.tar.gz -f
        obsutil cp obs://openpi-rtc/openpi05/inference_bundle.tar.gz . -f
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from obs import ObsClient

# =============================================================================
# ===== 在这里填写你的华为云 AK / SK（也可以改用环境变量 OBS_AK / OBS_SK）====
# =============================================================================
OBS_ACCESS_KEY_ID = ""  # TODO: 填入你的华为云 Access Key
OBS_SECRET_ACCESS_KEY = ""  # TODO: 填入你的华为云 Secret Key
# =============================================================================

DEFAULT_BUCKET = os.environ.get("OBS_BUCKET", "handzero-research")
DEFAULT_ENDPOINT = os.environ.get(
    "OBS_ENDPOINT", "https://obs.cn-north-4.myhuaweicloud.com"
)  # 北京区域（北京四 cn-north-4）
SINGLE_PUT_LIMIT = 64 * 1024 * 1024  # <=64MB 走单次 PUT
RESUMABLE_DOWNLOAD_LIMIT = 512 * 1024 * 1024  # >512MB 下载走断点续传
DEFAULT_PART_MB = 64


def _credentials() -> tuple[str, str]:
    ak = os.environ.get("OBS_AK") or OBS_ACCESS_KEY_ID
    sk = os.environ.get("OBS_SK") or OBS_SECRET_ACCESS_KEY
    if not ak or not sk:
        sys.exit(
            "ERROR: AK/SK 未配置。请在 obs_transfer.py 顶部填入 OBS_ACCESS_KEY_ID / "
            "OBS_SECRET_ACCESS_KEY，或设置环境变量 OBS_AK / OBS_SK。"
        )
    return ak.strip(), sk.strip()


def make_client(endpoint: str) -> ObsClient:
    ak, sk = _credentials()
    return ObsClient(
        access_key_id=ak,
        secret_access_key=sk,
        server=endpoint or DEFAULT_ENDPOINT,
    )


def parse_obs_ref(ref: str, bucket: str | None = None) -> tuple[str, str]:
    """ref 是 object key（可含前缀）；obs://bucket/key 可显式指定桶。"""
    bucket = bucket or DEFAULT_BUCKET
    ref = ref.strip()
    for scheme in ("obs://", "s3://"):
        if ref.startswith(scheme):
            rest = ref[len(scheme):]
            b, _, k = rest.partition("/")
            if not b or not k:
                sys.exit(f"ERROR: 无法解析 OBS 路径: {ref}")
            return b, k
    if not ref:
        sys.exit("ERROR: 空的 OBS 路径")
    return bucket, ref


def human(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def md5_of_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(resp, what: str) -> None:
    status = getattr(resp, "status", None)
    if status is None or int(status) >= 300:
        code = getattr(resp, "errorCode", "") or ""
        msg = getattr(resp, "errorMessage", "") or ""
        sys.exit(f"ERROR: {what} 失败 (status={status}): {code} {msg}".strip())


def _head_meta(client: ObsClient, bucket: str, key: str) -> tuple[int | None, str | None]:
    resp = client.headObject(bucket, key)
    _check(resp, f"head {bucket}/{key}")
    headers = {}
    for k, v in getattr(resp, "header", None) or []:
        headers[str(k).lower()] = v
    try:
        size = int(headers.get("content-length") or headers.get("contentlength"))
    except (TypeError, ValueError):
        size = None
    return size, headers.get("md5")


def _verify(client: ObsClient, bucket: str, key: str,
            expect_size: int, expect_md5: str | None = None) -> None:
    size, md5 = _head_meta(client, bucket, key)
    ok = size == expect_size
    print(f"  校验: size={human(size) if size is not None else '?'} "
          f"(期望 {human(expect_size)})", end="")
    if expect_md5 and md5:
        ok = ok and md5 == expect_md5
        print(f", md5={md5[:12]}… ({'一致' if md5 == expect_md5 else '不一致'})", end="")
    print(" -> " + ("OK" if ok else "FAILED"))
    if not ok:
        sys.exit("ERROR: 上传后校验失败")


def _upload_single(client: ObsClient, bucket: str, key: str, local: str) -> None:
    size = os.path.getsize(local)
    md5_hex = md5_of_file(local)
    print(f"单次 PUT: {local} ({human(size)}) -> {bucket}/{key}")
    resp = client.putFile(
        bucket,
        key,
        local,
        metadata={"md5": md5_hex},
        progressCallback=lambda read, total: None,
    )
    _check(resp, "上传")
    _verify(client, bucket, key, size, md5_hex)


def _upload_multipart(client: ObsClient, bucket: str, key: str, local: str,
                      part_mb: int, workers: int, force: bool) -> None:
    size = os.path.getsize(local)
    md5_hex = md5_of_file(local)
    print(f"分片上传: {local} ({human(size)}) -> {bucket}/{key} "
          f"(part={part_mb}MB, workers={workers}, 断点续传)")
    checkpoint = local + ".obs-upload.cp"
    if force:
        try:
            os.remove(checkpoint)
        except OSError:
            pass
    resp = client.uploadFile(
        bucket,
        key,
        local,
        partSize=part_mb * 1024 * 1024,
        taskNum=workers,
        enableCheckpoint=True,
        checkpointFile=checkpoint,
        metadata={"md5": md5_hex},
        progressCallback=lambda read, total: None,
    )
    _check(resp, "分片上传")
    _verify(client, bucket, key, size, md5_hex)
    try:
        os.remove(checkpoint)
    except OSError:
        pass


def upload(local: str, ref: str, part_mb: int, workers: int, force: bool,
           bucket: str, endpoint: str) -> None:
    if not os.path.isfile(local):
        sys.exit(f"ERROR: 文件不存在: {local}")
    bucket, key = parse_obs_ref(ref, bucket)
    client = make_client(endpoint)
    try:
        if os.path.getsize(local) <= SINGLE_PUT_LIMIT:
            _upload_single(client, bucket, key, local)
        else:
            _upload_multipart(client, bucket, key, local, part_mb, workers, force)
    finally:
        client.close()


def download(ref: str, local: str, bucket: str, endpoint: str) -> None:
    bucket, key = parse_obs_ref(ref, bucket)
    client = make_client(endpoint)
    try:
        size, _ = _head_meta(client, bucket, key)
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        print(f"下载: {bucket}/{key} ({human(size) if size is not None else '?'}) -> {local}")

        last = [0.0]

        def progress(read: int, total: int) -> None:
            now = time.time()
            if total and (now - last[0] >= 1.0 or read >= total):
                last[0] = now
                print(f"\r  {read * 100 / total:.1f}% ({human(read)}/{human(total)})",
                      end="", flush=True)

        if size is not None and size > RESUMABLE_DOWNLOAD_LIMIT:
            print("  (>512MB，使用断点续传下载)")
            resp = client.downloadFile(
                bucket,
                key,
                local,
                partSize=DEFAULT_PART_MB * 1024 * 1024,
                taskNum=4,
                enableCheckpoint=True,
                checkpointFile=local + ".obs-download.cp",
                progressCallback=progress,
            )
            _check(resp, "下载")
        else:
            resp = client.getObject(
                bucket, key, downloadPath=local, progressCallback=progress
            )
            _check(resp, "下载")
        print()
        got = os.path.getsize(local)
        if size is not None and got != size:
            sys.exit(f"ERROR: 下载大小不匹配: {got} != {size}")
        print(f"下载完成: {human(got)}")
    finally:
        client.close()


def ls(ref: str | None, bucket: str, endpoint: str) -> None:
    prefix = ref or ""
    client = make_client(endpoint)
    try:
        total = 0
        count = 0
        marker = None
        print(f"{bucket}/{prefix or ''}:")
        while True:
            resp = client.listObjects(bucket, prefix=prefix, marker=marker, max_keys=1000)
            _check(resp, "ls")
            contents = getattr(resp.body, "contents", None) or []
            for item in contents:
                total += int(item.size)
                count += 1
                print(f"  {item.key}  {human(item.size)}")
            if not getattr(resp.body, "is_truncated", False):
                break
            marker = getattr(resp.body, "next_marker", None)
        print(f"共 {count} 个对象, {human(total)}")
    finally:
        client.close()


def rm(ref: str, yes: bool, bucket: str, endpoint: str) -> None:
    bucket, key = parse_obs_ref(ref, bucket)
    if not yes:
        sys.exit("ERROR: 删除需要显式确认: 加 --yes")
    client = make_client(endpoint)
    try:
        resp = client.deleteObject(bucket, key)
        _check(resp, "删除")
        print(f"已删除: {bucket}/{key}")
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OBS 传输助手（openpi_rtc 推理 bundle，工控机通道）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ls = sub.add_parser("ls", help="列出对象")
    p_ls.add_argument("ref", nargs="?", help="前缀，如 openpi05/（默认列出整个桶）")
    p_ls.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"桶名（默认 {DEFAULT_BUCKET}）")
    p_ls.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p_ls.set_defaults(func=ls)

    p_up = sub.add_parser("upload", help="上传文件（大文件自动分片+断点续传）")
    p_up.add_argument("local", help="本地文件路径")
    p_up.add_argument("ref", help="object key（可含 openpi05/ 前缀）")
    p_up.add_argument("--part-mb", type=int, default=DEFAULT_PART_MB,
                      help=f"分片大小 MB（默认 {DEFAULT_PART_MB}）")
    p_up.add_argument("--workers", type=int, default=4, help="并发分片数（默认 4）")
    p_up.add_argument("--force", action="store_true", help="放弃旧任务重新上传")
    p_up.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"桶名（默认 {DEFAULT_BUCKET}）")
    p_up.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p_up.set_defaults(func=upload)

    p_dl = sub.add_parser("download", help="下载对象到本地文件")
    p_dl.add_argument("ref", help="object key（可含 openpi05/ 前缀）")
    p_dl.add_argument("local", help="本地保存路径")
    p_dl.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"桶名（默认 {DEFAULT_BUCKET}）")
    p_dl.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p_dl.set_defaults(func=download)

    p_rm = sub.add_parser("rm", help="删除对象（需 --yes）")
    p_rm.add_argument("ref", help="object key（可含 openpi05/ 前缀）")
    p_rm.add_argument("--yes", action="store_true", help="确认删除")
    p_rm.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"桶名（默认 {DEFAULT_BUCKET}）")
    p_rm.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p_rm.set_defaults(func=rm)

    args = parser.parse_args()
    if args.command == "ls":
        args.func(args.ref, args.bucket, args.endpoint)
    elif args.command == "upload":
        args.func(args.local, args.ref, args.part_mb, args.workers, args.force,
                  args.bucket, args.endpoint)
    elif args.command == "download":
        args.func(args.ref, args.local, args.bucket, args.endpoint)
    elif args.command == "rm":
        args.func(args.ref, args.yes, args.bucket, args.endpoint)


if __name__ == "__main__":
    main()
