from data.schemas import SeqBatch


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def batch_to(batch, device):
    user_ids = batch.user_ids
    ids = batch.ids
    ids_fut = batch.ids_fut
    x = batch.x
    x_fut = batch.x_fut
    seq_mask = batch.seq_mask
    return SeqBatch(*[v.to(device) for _,v in batch._asdict().items()])

def next_batch(dataloader, device):
    # print(dataloader)
    batch = next(dataloader)
    return batch_to(batch, device)
