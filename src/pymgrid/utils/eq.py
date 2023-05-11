import numpy as np


def verbose_eq(obj_a, obj_b, attrs_or_indices, indent=0):
    if obj_a == obj_b:
        print('Objects are equal.')
        return

    reason_num = 1
    ind = '\t' * indent

    def print_reason(reason, num):
        print(f"{ind}{num}) {reason}")
        num += 1
        return num

    if type(obj_a) != type(obj_b):
        print_reason(f'type(self)={type(obj_a)} != type(other)={type(obj_b)}"', reason_num)
        return

    print(f'{ind}{type(obj_a).__name__}s are not equal due to: ')

    try:
        if len(obj_a) != len(obj_b):
            reason_num = print_reason(f'len(self)={len(obj_a)} != len(other)={len(obj_b)}"', reason_num)
    except TypeError:
        pass

    for attr_idx in attrs_or_indices:
        self_attr, other_attr, reason = _get_attrs_reason(obj_a, obj_b, attr_idx)

        try:
            eq = bool(self_attr == other_attr)
        except ValueError:
            eq = np.array_equal(self_attr, other_attr)

        if not eq:
            reason_num = print_reason(reason, reason_num)
            try:
                self_attr.verbose_eq(other_attr, indent=indent+1)
            except AttributeError:
                pass


def _get_attrs_reason(obj_a, obj_b, attr_idx):
    if isinstance(attr_idx, int):
        self_attr = obj_a[attr_idx]
        other_attr = obj_b[attr_idx]
        reason = f'self[{attr_idx}] != other[{attr_idx}]'

    else:
        self_attr = getattr(obj_a, attr_idx)
        other_attr = getattr(obj_b, attr_idx)
        reason = f'self.{attr_idx} != other.{attr_idx}'

    return self_attr, other_attr, reason
