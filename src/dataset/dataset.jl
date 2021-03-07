"""
    AbstractDataset

Abstract type for dataset objects.
"""
abstract type AbstractDataset end

function download(dataset::AbstractDataset)
    error("download method is not implemented.")
end

function load(dataset::AbstractDataset)
    error("load method is not implemented.")
end