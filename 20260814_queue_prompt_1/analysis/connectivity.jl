using DataStructures  # for IntDisjointSets


function check_connectivity(rankings, n_items)
    uf = IntDisjointSets(n_items)
    for π in rankings, i in 1:length(π)-1, j in i+1:length(π)
        union!(uf, π[i], π[j])
    end
    roots = [find_root!(uf, i) for i in 1:n_items]
    n_components = length(unique(roots))
    return n_components, roots
end

n_components, roots = check_connectivity(rankings, 4000)
