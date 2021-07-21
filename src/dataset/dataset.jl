"""
    AbstractDataset

Abstract type for dataset objects.
"""
abstract type AbstractDataset end

function download(dataset::AbstractDataset)
    error("download method is not implemented.")
end

function load_dataset(dataset::AbstractDataset)
    error("load_dataset method is not implemented.")
end