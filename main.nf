nextflow.enable.dsl=2

// Helper functions

def get_single_fastx( glob ){
    return channel.fromPath(glob) | map { file -> tuple(file.baseName, file) }
}

def get_paired_fastq( glob ){
    return channel.fromFilePairs(glob, flat: true)
}

def get_dir( glob ){
    return channel.fromPath(glob, type: 'dir').map { tuple(it.getName(), it) }
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

// GraftM search workflow
// nextflow run np-core/druid --container np-core/graftm -profile docker --workflow graftm_search --fastq "*_{1,2}.fastq"--packages /path/to/packages --outdir graftm_search

params.packages = ""  // directory with graftm packages


// MAG assembly workflow

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
    main:
        GraftM(reads, packages)
}


workflow dnd {
   if (params.workflow == "mag_assembly") {
       get_paired_fastq(params.fastq) | metawrap_assembly
   }
   
   if (params.workflow == "mag_search"){
        get_single_fastx(params.fasta) | view
   }
   if (params.workflow == "graftm_search"){
        reads = get_paired_fastq(params.fastq)
        packages = get_dir(params.packages)

        graftm_search(reads.combine(packages))
   }
}

// Execute

workflow {
    dnd()
}