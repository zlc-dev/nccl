"""Summarize ZIP_SLICE intervals for the last invocation on each rank/channel."""
import argparse
from collections import defaultdict


def read_last_call(lines):
    channels = {}
    for line in lines:
        if not line.startswith("ZIP_SLICE "):
            continue
        fields = dict(token.split("=") for token in line.split()[1:])
        key = int(fields["rank"]), int(fields["channel"])
        row = int(fields["row"])
        if row == 0:
            channels[key] = []
        records = channels[key]
        if row != len(records):
            raise ValueError(f"missing or unordered row on rank/channel {key}")
        records.append(fields)
    if not channels:
        raise ValueError("no ZIP_SLICE records found")
    for key, records in channels.items():
        if len(records) != min(8, int(records[0]["total"])):
            raise ValueError(f"incomplete trace on rank/channel {key}")
    return channels


def report(rank, channels):
    # Use only timestamps from this GPU. Merge overlapping recv/send polls on
    # the same channel, even when both roles poll simultaneously.
    events = defaultdict(list)
    for channel, records in channels.items():
        for record in records:
            for field in ("recv", "send", "codec"):
                start, end = map(int, record[field].split(","))
                if end < start:
                    raise ValueError(f"invalid {field} interval on rank {rank}")
                if end > start:
                    kind = "codec" if field == "codec" else "poll"
                    events[start].append((kind, channel, 1))
                    events[end].append((kind, channel, -1))
    active = {"codec": defaultdict(int), "poll": defaultdict(int)}
    wait = covered = 0
    previous = 0
    for time in sorted(events):
        elapsed = time - previous
        for channel in active["poll"]:
            wait += elapsed
            if len(active["codec"]) > int(channel in active["codec"]):
                covered += elapsed
        for kind, channel, delta in events[time]:
            counts = active[kind]
            counts[channel] += delta
            if counts[channel] == 0:
                del counts[channel]
        previous = time
    fraction = f"{100 * covered / wait:.1f}%" if wait else "n/a"
    print(f"rank={rank} channels={len(channels)} poll_channel_us={wait/1000:.2f} "
          f"overlap_channel_us={covered/1000:.2f} overlap={fraction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    args = parser.parse_args()
    by_rank = defaultdict(dict)
    with open(args.log) as log:
        for (rank, channel), records in read_last_call(log).items():
            by_rank[rank][channel] = records
    for rank in sorted(by_rank):
        report(rank, by_rank[rank])
