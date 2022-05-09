
class AbstractBlockBuilding:


    is_using_entropy = False
    num_of_entities_d1 = None
    num_of_entities_d2 = None

    blocks = list()
    entity_profiles_d1 = list()
    entity_profiles_d2 = list()
    inverted_index_d1 = dict()
    inverted_index_d2 = dict()
    schema_clusters = list()


    def __init__(self):
        self.is_using_cross_entropy = False

    def build_blocks(self):

        if self.schema_clusters:
            index_entities(self.inverted_index_d1, self.entity_profiles_d1, self.schema_clusters[0])
        else:
            index_entities(self.inverted_index_d1, self.entity_profiles_d1)


        if self.inverted_index_d2:
            if self.schema_clusters:    
                index_entities(self.inverted_index_d1, self.entity_profiles_d1, self.schema_clusters[0])
            else:
                index_entities(self.inverted_index_d1, self.entity_profiles_d1)
