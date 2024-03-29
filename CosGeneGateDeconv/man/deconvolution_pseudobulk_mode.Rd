% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/CosGeneGateDeconvPBMode.R
\name{deconvolution_pseudobulk_mode}
\alias{deconvolution_pseudobulk_mode}
\title{This is the function for CosGeneGate deconvolution - pseudobulk mode}
\usage{
deconvolution_pseudobulk_mode(
  sc_profile,
  mkg_file,
  index = c(2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50),
  prop = 0.9,
  n = 50,
  plot_line = FALSE
)
}
\arguments{
\item{sc_profile}{File directory for single cell reference file.}

\item{mkg_file}{Marker gene file generated by CosGeneGate.}

\item{index}{Numbe of marker genes to be iterated.}

\item{prop}{Proportion of cells remained in the reference profile. The rest of the cells will be used to generate pseudobulk mixtures.}

\item{n}{Number of pseudobulk mixtures to be generated.}

\item{plot_line}{TRUE to plot the graph of error vs. # of marker genes, FALSE otherwise.}
}
\value{
Return the optimized single cell reference profile.
}
\description{
This is the function for CosGeneGate deconvolution - pseudobulk mode
}
