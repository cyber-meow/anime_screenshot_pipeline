from dataclasses import dataclass
import json



class TagTree:
    class Node:
        __slots__ = ['name', 'parent', 'childs', 'is_tag']
        def __init__(
            self,
            name: str = '',
            is_tag: bool = False,
            parent = None,
            childs = {},
        ) -> None:
            self.name = name
            self.is_tag = is_tag
            self.parent = parent
            self.childs = childs
    
    def __init__(self) -> None:
        self.node_table: dict[str, TagTree.Node] = {}
        self.root = self.Node('root', False, None, [])
    
    def build_from_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.build_from_dict(data)
    
    def _build_from_dict(self, root, key, data: dict|str):
        if data is None:
            return None
        if isinstance(data, str):
            return TagTree.Node(key, True, root, {})
        self_data = data.pop('self', None)
        
        is_tag = isinstance(self_data, str)
        new_node = TagTree.Node(
            key, is_tag, root
        )
        
        all_childs = {}
        if isinstance(self_data, dict):
            self_node = TagTree.Node(f'{root.name}-self', 'self')
            # all_childs[key] = self._build_from_dict(root, k, v)
        for k, v in data.items():
            if v is None: continue
            child = self._build_from_dict(new_node, k, v)
            self.node_table[k] = self.node_table.get(k, []) + [child]
            all_childs[k] = child
        
        new_node.childs = all_childs
        return new_node
    
    def build_from_dict(self, data: dict):
        all_childs = {}
        for k, v in data.items():
            if v is None: continue
            child = self._build_from_dict(self.root, k, v)
            self.node_table[k] = self.node_table.get(k, []) + [child]
            all_childs[k] = child
        
        self.root.childs = all_childs
    
    def find_nodes(self, query: list[str], reverse_query=False):
        query = list(query)
        if reverse_query:
            query.reverse()
        query_root = query.pop(0)
        if query_root not in self.node_table:
            raise ValueError('Tag/Groups not Found !')
        target_node = self.node_table[query_root]
        if len(target_node)>1:
            raise ValueError(
                'Have multiple groups with same name, '
                'please give some parent group for querying'
            )
        
        target_node = target_node[0]
        for i in query:
            if i not in target_node.childs:
                raise ValueError('Tag/Groups not Found !')
            target_node = target_node.childs[i]
        return target_node
    
    def get_groups(self, query, reverse_query=False):
        if isinstance(query, str):
            query = [query]
        
        target_node = self.find_nodes(query, reverse_query)
        all_groups = [target_node.name]
        while target_node.parent is not None:
            target_node = target_node.parent
            all_groups.append(target_node.name)
        
        return all_groups
    
    def _get_tags(self, node: 'TagTree.Node'):
        res = []
        if node.is_tag:
            res = [node.name]
        for i in node.childs.values():
            res += self._get_tags(i)
        return res
    
    def get_tags(self, query, reverse_query=False):
        if isinstance(query, str):
            query = [query]
        target_node = self.find_nodes(query, reverse_query)
        all_tag = self._get_tags(target_node)
        return all_tag


tree = TagTree()
tree.build_from_json('./tag_tree.json')

tag = ['Attire']
print(f'query tag/groups: {tag}')
print(tree.get_tags(tag, reverse_query=True)[:10])
print(tree.get_groups(tag))