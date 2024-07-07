# configuration like t.ex-graph-2.0-classifier for fair comparison

excluded_features = [
    'id', 
    'tracking',
    'binary_tracker',
    'multi_tracker'
] + [
    'ping',
    'delete',
    'search',
    'patch',
    'websocket'
] + [
    'label',
    'timeset',
    'weighted indegree',
    'weighted outdegree',
    'weighted degree',
]