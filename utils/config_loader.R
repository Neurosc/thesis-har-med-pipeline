library(RcppTOML)  # if missing: install.packages("RcppTOML")

REPO_ROOT   <- normalizePath(file.path(dirname(sys.frame(1)$ofile), ".."))
CONFIG_PATH <- file.path(REPO_ROOT, "config.toml")

load_config <- function() {
  cfg <- parseTOML(CONFIG_PATH)
  list(
    denoising_method = cfg$active$denoising_method,
    atlas_method     = cfg$active$atlas_method
  )
}

tag <- function() {
  c <- load_config()
  paste0(c$atlas_method, "_", c$denoising_method)
}
