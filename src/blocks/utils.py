def create_entity_index(blocks: dict, is_dirty_er: bool):
    '''
     Creates a dict of entity ids -> block ids
    '''
    # TODO remove entities
    num_of_entities_1 = 0
    num_of_entities_2 = 0
    num_of_blocks = 0

    entity_index = {}
    for key, block in blocks.items():
        for entity_id in block.entities_D1:
            entity_index.setdefault(entity_id, [])
            entity_index[entity_id].append(key)
            num_of_entities_1 += 1

        if not is_dirty_er:
            for entity_id in block.entities_D2:
                entity_index.setdefault(entity_id, [])
                entity_index[entity_id].append(key)
                num_of_entities_2 += 1
        num_of_blocks += 1

    return entity_index, {
        'num_of_blocks' : num_of_blocks,
        'num_of_entities_1' : num_of_entities_1,
        'num_of_entities_2' : num_of_entities_2
        }

def drop_single_entity_blocks(blocks: dict, is_dirty_er: bool) -> dict:
    '''
     Removes one-size blocks for DER and empty for CCER
    '''
    all_keys = list(blocks.keys())
    # print("All keys before: ", len(all_keys))
    if is_dirty_er:
        for key in all_keys:
            if len(blocks[key].entities_D1) == 1:
                blocks.pop(key)
    else:
        for key in all_keys:
            if len(blocks[key].entities_D1) == 0 or len(blocks[key].entities_D2) == 0:
                blocks.pop(key)

    return blocks
