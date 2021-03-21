
// @JCU: nextflow np-core/druid -profile zodiac --container ~/bin/metawrap.sif --worfklow mag_assembly --fastq "/path/to/*_{1,2}.fastq" --outdir test_assembly --completeness 70 --contamination 5 --assembly_memory 300G --process_memory 128G --assembly_threads 80 --process_threads 32

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
        } else {
            null
        }
    }
    .filter { it != null }

}

params.workflow = 'mag_assembly'

params.fastq = "*_{1,2}.fastq"
params.outdir = 'mag_assembly'

params.qc_options = '--skip-bmtagger'
params.assembly_options = '--metaspades'

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

workflow graftm_search {

    take:
        reads  // id, fwd, rv
        packages // id, dir
    main:
        GraftM(reads, packages)
}


workflow dnd {
   if (params.workflow == "mag_assembly") {
       get_paired_fastq(params.fastq) | metawrap_assembly
   }
   
   if (params.workflow == "graftm_search"){
        get_paired_fastq(params.fastq) | view
   }
   if (params.workflow == "mag_search"){
        graftm_search(get_single_fastx(params.fastq), get_directories(params.packages))
   }
}

// Execute

workflow {
    dnd()
}