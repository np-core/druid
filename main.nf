// Hybrid assembly workflow: DSL2

nextflow.enable.dsl=2

// Helper functions

def get_single_fastx( glob ){
    return channel.fromPath(glob) | map { file -> tuple(file.baseName, file) }
}

def get_paired_fastq( glob ){
    return channel.fromFilePairs(glob, flat: true)
}

def get_matching_data( channel1, channel2){

    // Get matching data by ID (first field) from two channels
    // by crossing, checking for matched ID and returning
    // a combined data tuple with a single ID (first field)

    channel1.cross(channel2).map { crossed ->
        if (crossed[0][0] == crossed[1][0]){
                tuple( crossed[0][0], *crossed[0][1..-1], *crossed[1][1..-1] )
            }
        } else {
            null
        }
    }
    .filter { it != null }

}


params.workflow = 'mag_assembly'
params.outdir = 'mag_assembly'
params.qc_options = '--skip-bmtagger'
params.assembly_options = '--metaspades'
params.fastq = "*_{1,2}.fastq"
params.completeness = 70
params.contamination = 5

// Modules

include { MetaWrapQC  } from './modules/metawrap'
include { MetaWrapAssembly  } from './modules/metawrap'
include { MetaWrapBinning } from './modules/metawrap'
include { MetaWrapBinAssembly  } from './modules/metawrap'
include { MetaWrapBinOps  } from './modules/metawrap'

workflow metawrap_assembly {

    // Illumina PE MetaWRAP

    take:
        reads  // id, fwd, rv
    main:
        MetaWrapQC(reads)
        MetaWrapAssembly(MetaWrapQC.out)
        MetaWrapBinning(MetaWrapAssembly.out, MetaWrapQC.out)
        MetaWrapBinAssembly(MetaWrapBinning.out, MetaWrapQC.out)
        MetaWrapBinOps(MetaWrapBinAssembly.out, MetaWrapQC.out)
}


workflow mag_dnd {

   if (params.workflow == "mag_assembly") {
       get_paired_fastq(params.fastq) | metawrap_assembly
   }
   if (params.workflow == "mag_search"){
        get_single_fastx(params.fastq) | view
   }
}

// Execute

workflow {
    mag_dnd()
}