"""
    AbstractDataset

Abstract type for dataset objects.
"""
abstract type AbstractDataset end

function download(dataset::AbstractDataset)
    error("download method is not implemented.")
end

function load_all(dataset::AbstractDataset)
    error("load_all method is not implemented.")
end