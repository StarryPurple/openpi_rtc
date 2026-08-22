#!/usr/bin/env python3
"""BOS transfer helper for the openpi_rtc bundles.

Only depends on the official Baidu BOS SDK (bce-python-sdk). Install once:

    uv pip install bce-python-sdk==0.9.76
    # or: pip install bce-python-sdk==0.9.76

Usage (ref 是 object key，可带 openpi05/ 前缀；默认桶 handzero-research，可用 --bucket 换桶):
    python bos_transfer.py ls [prefix] [--bucket handzero-research] [--endpoint bj.bcebos.com]
    python bos_transfer.py upload <local_file> <ref> [--part-mb 64] [--workers 4] [--force]
    python bos_transfer.py download <ref> <local_file>
    python bos_transfer.py rm <ref> --yes

Examples (bucket defaults to handzero-research):
    # list everything under openpi05/
    python bos_transfer.py ls openpi05/

    # upload the training bundle + raw data
    python bos_transfer.py upload /tmp/openpi_training_bundle.tar.gz openpi05/training_bundle.tar.gz
    python bos_transfer.py upload /tmp/openpi_training_bundle.data.tar.gz openpi05/raw_data.tar.gz

    # upload the inference bundle
    python bos_transfer.py upload /tmp/openpi_inference_bundle.tar.gz openpi05/inference_bundle.tar.gz

    # pull them on the training / inference machine
    python bos_transfer.py download openpi05/training_bundle.tar.gz /tmp/training_bundle.tar.gz

Credentials:
    Fill ACCESS_KEY_ID / SECRET_ACCESS_KEY below, or set the BOS_AK / BOS_SK
    environment variables. The script never prints the credentials.

Notes:
    - Files below 64MB use a single PUT; larger files use multipart upload with
      resume. The resume state lives next to the local file in
      "<local_file>.bos-upload.json"; delete it (or pass --force) to restart.
    - After upload the object is verified: size and (if available) an md5 we
      attach as user metadata x-bce-meta-md5.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.bos.bos_client import BosClient

# =============================================================================
# ===== 在这里填写你的 AK / SK（也可以改用环境变量 BOS_AK / BOS_SK）=========
# =============================================================================
ACCESS_KEY_ID = ""  # TODO: 填入你的百度云 Access Key
SECRET_ACCESS_KEY = ""  # TODO: 填入你的百度云 Secret Key
# =============================================================================

DEFAULT_BUCKET = "handzero-research"
DEFAULT_ENDPOINT = "bj.bcebos.com"  # 北京区域
SINGLE_PUT_LIMIT = 64 * 1024 * 1024  # <=64MB 走单次 PUT
MAX_PARTS = 10000  # BOS 上限
PART_RETRIES = 3


def _credentials() -> tuple[str, str]:
    ak = os.environ.get("BOS_AK") or ACCESS_KEY_ID
    sk = os.environ.get("BOS_SK") or SECRET_ACCESS_KEY
    if not ak or not sk:
        sys.exit(
            "ERROR: AK/SK 未配置。请在 bos_transfer.py 顶部填入 ACCESS_KEY_ID / "
            "SECRET_ACCESS_KEY，或设置环境变量 BOS_AK / BOS_SK。"
        )
    return ak.strip(), sk.strip()


def make_client(endpoint: str) -> BosClient:
    ak, sk = _credentials()
    config = BceClientConfiguration(
        credentials=BceCredentials(ak, sk),
        endpoint=endpoint or DEFAULT_ENDPOINT,
    )
    return BosClient(config)


def parse_bos_ref(ref: str, bucket: str | None = None) -> tuple[str, str]:
    """ref 是 object key（可含前缀）；bos://bucket/key 可显式指定桶。"""
    bucket = bucket or DEFAULT_BUCKET
    ref = ref.strip()
    for scheme in ("bos://", "s3://", "bce://"):
        if ref.startswith(scheme):
            rest = ref[len(scheme):]
            b, _, k = rest.partition("/")
            if not b or not k:
                sys.exit(f"ERROR: 无法解析 BOS 路径: {ref}")
            return b, k
            break
    if not ref:
        sys.exit("ERROR: 空的 BOS 路径")
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


class _Progress:
    """Thread-safe progress counter; prints 'part done / total' lines only."""

    def __init__(self, total_parts: int):
        self.total_parts = total_parts
        self.done = 0
        self.done_bytes = 0
        self.lock = threading.Lock()

    def add(self, part_bytes: int, part_number: int):
        with self.lock:
            self.done += 1
            self.done_bytes += part_bytes
            print(
                f"  part {part_number}/{self.total_parts} done "
                f"({self.done}/{self.total_parts}, {human(self.done_bytes)})"
            )


def _head_meta(client: BosClient, bucket: str, key: str) -> tuple[int | None, str | None]:
    meta = client.get_object_meta_data(bucket, key).metadata
    size = meta.content_length
    try:
        size = int(size)
    except (TypeError, ValueError):
        size = None
    return size, meta.x_bce_meta_md5


def _upload_single(client: BosClient, bucket: str, key: str, local: str) -> None:
    size = os.path.getsize(local)
    md5_hex = md5_of_file(local)
    print(f"单次 PUT: {local} ({human(size)}) -> {bucket}/{key}")
    client.put_object_from_file(
        bucket,
        key,
        local,
        content_md5=base64.b64encode(bytes.fromhex(md5_hex)).decode(),
        user_metadata={"md5": md5_hex},
        progress_callback=lambda read, total: None,
    )
    _verify(client, bucket, key, size, md5_hex)


def _verify(client: BosClient, bucket: str, key: str, expect_size: int, expect_md5: str) -> None:
    size, md5 = _head_meta(client, bucket, key)
    ok = size == expect_size
    print(f"  校验: size={human(size) if size is not None else '?'} "
          f"(期望 {human(expect_size)})", end="")
    if md5:
        ok = ok and md5 == expect_md5
        print(f", md5={md5[:12]}… ({'一致' if md5 == expect_md5 else '不一致'})", end="")
    print(" -> " + ("OK" if ok else "FAILED"))
    if not ok:
        sys.exit("ERROR: 上传后校验失败")


def _manifest_path(local: str) -> str:
    return local + ".bos-upload.json"


def _load_manifest(client: BosClient, local: str, bucket: str, key: str,
                   size: int, md5_hex: str) -> tuple[str | None, dict[str, str]]:
    """Return (upload_id, {part_number: etag}) for resuming, or (None, {})."""
    mp = _manifest_path(local)
    if not os.path.exists(mp):
        return None, {}
    try:
        with open(mp, "r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        return None, {}
    if (m.get("bucket"), m.get("key")) != (bucket, key):
        return None, {}
    if m.get("local_size") != size or m.get("local_md5") != md5_hex:
        print("本地文件已变化，重新开始分片上传")
        return None, {}
    upload_id = m.get("upload_id")
    if not upload_id:
        return None, {}
    # 以服务端为准：把已上传完成的分片捞回来
    server_parts: dict[str, str] = {}
    try:
        for p in client.list_all_parts(bucket, key, upload_id):
            server_parts[str(p.part_number)] = p.etag
    except Exception as e:
        print(f"查询已有分片失败（{e}），重新开始")
        return None, {}
    if not server_parts:
        return None, {}
    print(f"恢复分片上传: upload_id={upload_id[:12]}…, 已上传 {len(server_parts)} 片")
    return upload_id, server_parts


def _save_manifest(local: str, bucket: str, key: str, upload_id: str,
                   size: int, md5_hex: str, parts: dict[str, str]) -> None:
    m = {
        "bucket": bucket,
        "key": key,
        "upload_id": upload_id,
        "local_size": size,
        "local_md5": md5_hex,
        "parts": parts,
    }
    tmp = _manifest_path(local) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    os.replace(tmp, _manifest_path(local))


def _upload_multipart(client: BosClient, bucket: str, key: str, local: str,
                      part_mb: int, workers: int, force: bool) -> None:
    size = os.path.getsize(local)
    md5_hex = md5_of_file(local)
    part_size = part_mb * 1024 * 1024
    total_parts = (size + part_size - 1) // part_size
    if total_parts > MAX_PARTS:
        sys.exit(f"ERROR: 分片数 {total_parts} 超过上限 {MAX_PARTS}，请调大 --part-mb")
    print(f"分片上传: {local} ({human(size)}, {total_parts} 片 x {human(part_size)}) "
          f"-> {bucket}/{key}")

    upload_id, done_parts = _load_manifest(client, local, bucket, key, size, md5_hex)
    if upload_id and force:
        print("--force: 放弃旧上传任务")
        try:
            client.abort_multipart_upload(bucket, key, upload_id)
        except Exception:
            pass
        upload_id, done_parts = None, {}
    if not upload_id:
        upload_id = client.initiate_multipart_upload(bucket, key).upload_id
        done_parts = {}
        print(f"新任务: upload_id={upload_id[:12]}…")
        _save_manifest(local, bucket, key, upload_id, size, md5_hex, done_parts)

    progress = _Progress(total_parts)
    lock = threading.Lock()

    def upload_part(part_number: int) -> tuple[int, str]:
        offset = (part_number - 1) * part_size
        part_len = min(part_size, size - offset)
        last_err = None
        for attempt in range(1, PART_RETRIES + 1):
            try:
                resp = client.upload_part_from_file(
                    bucket, key, upload_id, part_number, part_len, local, offset
                )
                return part_number, resp.metadata.etag
            except Exception as e:
                last_err = e
                if attempt < PART_RETRIES:
                    time.sleep(2 * attempt)
        raise RuntimeError(f"part {part_number} 上传失败: {last_err}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for n in range(1, total_parts + 1):
            if str(n) in done_parts:
                progress.add(part_size if n < total_parts else size - (n - 1) * part_size, n)
            else:
                futures[pool.submit(upload_part, n)] = n
        for fut in as_completed(futures):
            n = futures[fut]
            etag = fut.result()
            part_len = min(part_size, size - (n - 1) * part_size)
            with lock:
                done_parts[str(n)] = etag
                _save_manifest(local, bucket, key, upload_id, size, md5_hex, done_parts)
            progress.add(part_len, n)

    part_list = sorted(
        ({"partNumber": int(n), "eTag": etag} for n, etag in done_parts.items()),
        key=lambda x: x["partNumber"],
    )
    if len(part_list) != total_parts:
        sys.exit(f"ERROR: 分片不齐 {len(part_list)}/{total_parts}，请重跑以断点续传")

    print("合并分片 (complete_multipart_upload) ...")
    client.complete_multipart_upload(
        bucket, key, upload_id, part_list, user_metadata={"md5": md5_hex}
    )
    _verify(client, bucket, key, size, md5_hex)
    try:
        os.remove(_manifest_path(local))
    except OSError:
        pass


def upload(local: str, ref: str, part_mb: int, workers: int, force: bool,
           bucket: str, endpoint: str) -> None:
    if not os.path.isfile(local):
        sys.exit(f"ERROR: 文件不存在: {local}")
    bucket, key = parse_bos_ref(ref, bucket)
    client = make_client(endpoint)
    if os.path.getsize(local) <= SINGLE_PUT_LIMIT:
        _upload_single(client, bucket, key, local)
    else:
        _upload_multipart(client, bucket, key, local, part_mb, workers, force)


def download(ref: str, local: str, bucket: str, endpoint: str) -> None:
    bucket, key = parse_bos_ref(ref, bucket)
    client = make_client(endpoint)
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    size, _ = _head_meta(client, bucket, key)
    print(f"下载: {bucket}/{key} ({human(size) if size is not None else '?'}) -> {local}")

    last = [0.0]

    def progress(read: int, total: int) -> None:
        now = time.time()
        if total and (now - last[0] >= 1.0 or read >= total):
            last[0] = now
            pct = read * 100 / total
            print(f"\r  {read / total * 100:.1f}% ({human(read)}/{human(total)})", end="", flush=True)

    try:
        client.get_object_to_file(bucket, key, local, progress_callback=progress)
    finally:
        print()
    got = os.path.getsize(local)
    if size is not None and got != size:
        sys.exit(f"ERROR: 下载大小不匹配: {got} != {size}")
    print(f"下载完成: {human(got)}")


def ls(ref: str | None, bucket: str, endpoint: str) -> None:
    prefix = ref or ""
    client = make_client(endpoint)
    total = 0
    count = 0
    print(f"{bucket}/{prefix or ''}:")
    for item in client.list_all_objects(bucket, prefix=prefix):
        total += int(item.size)
        count += 1
        print(f"  {item.key}  {human(item.size)}")
    print(f"共 {count} 个对象, {human(total)}")


def rm(ref: str, yes: bool, bucket: str, endpoint: str) -> None:
    bucket, key = parse_bos_ref(ref, bucket)
    if not yes:
        sys.exit("ERROR: 删除需要显式确认: 加 --yes")
    client = make_client(endpoint)
    client.delete_object(bucket, key)
    print(f"已删除: {bucket}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BOS 传输助手（openpi_rtc 训练/推理 bundle）",
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
    p_up.add_argument("--part-mb", type=int, default=64, help="分片大小 MB（>=5，默认 64）")
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
        args.func(args.local, args.ref, args.part_mb, args.workers, args.force, args.bucket, args.endpoint)
    elif args.command == "download":
        args.func(args.ref, args.local, args.bucket, args.endpoint)
    elif args.command == "rm":
        args.func(args.ref, args.yes, args.bucket, args.endpoint)


if __name__ == "__main__":
    main()
