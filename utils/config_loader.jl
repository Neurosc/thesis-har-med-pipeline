using TOML

const REPO_ROOT   = abspath(joinpath(@__DIR__, ".."))
const CONFIG_PATH = joinpath(REPO_ROOT, "config.toml")

function load_config()
    cfg = TOML.parsefile(CONFIG_PATH)
    return (
        denoising_method = cfg["active"]["denoising_method"],
        atlas_method     = cfg["active"]["atlas_method"],
    )
end

function tag()
    c = load_config()
    return "$(c.atlas_method)_$(c.denoising_method)"
end
