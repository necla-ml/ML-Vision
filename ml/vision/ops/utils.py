
## Detection manipulation
def dets_select(dets, inclusion, inverse=False):
    """
    Args:
        dets(xyxysc, List[xyxysc]): detection tensor(s)
        inclusion(List[int]): list of classes to include
        inverse(bool): invert the selection or not
    """
    res = []
    listed = isinstance(dets, list)
    if not listed:
        dets = [dets]
    for i, dets_i in enumerate(dets):
        selection = dets_i[:, -1] == -1
        for p in inclusion:
            selection |= dets_i[:, -1] == p
        res.append(~selection if inverse else selection)
    return res if listed else res[0]

