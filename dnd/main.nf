/ Hybrid assembly workflow: DSL2

nextflow.enable.dsl=2

// Helper functions

def get_single_fastx( glob ){
    return channel.fromPath(glob) | map { file -> tuple(file.baseName, file) }
}

def get_paired_fastq( glob ){
    return channel.fromFilePairs(glob, flat: true)
}

def get_matching_data( channel1, channel2, illumina = false ){

    // Get matching data by ID (first field) from two channels
    // by crossing, checking for matched ID and returning
    // a combined data tuple with a single ID (first field)

    channel1.cross(channel2).map { crossed ->
        if (crossed[0][0] == crossed[1][0]){
            if (illumina){
                // First channel returning Illumina only for hybrid reference assembly
                crossed[0]
            } else {
                // Return mix of channels for evaluations in hybrid assembly workflow
                tuple( crossed[0][0], *crossed[0][1..-1], *crossed[1][1..-1] )
            }
        } else {
            null
        }
    }
    .filter { it != null }

}

// ONT Quality control subworkflow


params.workflow = 'hybrid'
params.reference = ""
params.fastq = 'fastq/*.fq'
params.outdir = 'results'
params.length = 200
params.quality = 7
params.coverage = 200
params.genome_size = '2.8m'
params.preset = 'minimap2-ont'  // coverm

// ONT assembly and polishing subworkflow

params.assembly_options = "--plasmids"
params.medaka_model = "r941_min_high_g360"

if ( file(params.medaka_model).exists() ){
    params.medaka_model = file(params.medaka_model)
}

// Illumina assembly, hybrid correction and reference comparison subworkflow

params.fasta = "fasta/*.fasta"
params.illumina = "fastq/*_R{1,2}.fastq.gz"
params.depth = 200
params.assembler = "skesa"

params.tag = null
params.saureus = true
params.kpneumoniae = false

// Stage files

reference = file(params.reference)

// Modules

include { Fastp } from '../modules/fastp'
include { Rasusa } from './modules/rasusa'
include { CoverM  } from './modules/coverm'

workflow clusterDnD {
    take:
        reads // id, reads
    main:

    emit:

}



workflow dnd_enquiry {

   if (params.workflow == "search_assemblies") {
       get_single_fastx(params.fasta)  | clusterDnD
   }
   if (params.workflow == "search_metagenomes"){
        get_single_fastx(params.fastq)
   }
}

// Execute

workflow {
    dnd_enquiry()
}