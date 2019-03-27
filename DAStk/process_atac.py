#! /usr/bin/env/python

from __future__ import print_function
import argparse
import csv
import datetime
import glob
import multiprocessing
import numpy as np
import os.path
import pandas as pd
import sys
import matplotlib as mpl
import pybedtools
# to prevent DISPLAY weirdness when running in the cluster:
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from functools import partial
from operator import itemgetter
#from functools import reduce


# returns ith column from the given matrix
def get_column(matrix,i):
    f = itemgetter(i)
    return map(f,matrix)


# return whether the given interval is within a window of size window_size
# around the given mean
def is_in_window(motif_interval, atac_median, window_size):
    start = int(motif_interval.start)
    end = int(motif_interval.end)
    if (end >= (atac_median - window_size) and end <= (atac_median + window_size)) \
        or (start >= (atac_median - window_size) and start <= (atac_median + window_size)):
        return True
    else:
        return False

def makebed12(finaldf, tf_motif_filename, atac_peaks_filename):
    atac_df = pd.read_csv(atac_peaks_filename, header=None, comment='#', sep="\t", na_filter=False)
    motif_df = pd.read_csv(tf_motif_filename, header=None, comment='#', sep="\t")
    atac_cols = len(atac_df.columns)
    motif_cols = len(motif_df.columns)
    print ("atac header", len(atac_df.columns))
    #step 1 pull in the atac and motif files and make them 6 column bed files, even if they are not 6 column bed files
    if atac_cols==3:
        atac_df = pd.read_csv(atac_peaks_filename, header=None, comment='#', sep="\t", na_filter=False, usecols=[0, 1, 2],names=["atac.chrom","atac_peak.start","atac_peak.end"])
        atac_df['atac_peak.name']='.'    
        atac_df['atac_peak.strand']='.'
        atac_df['atac_peak.score']='.'
    if atac_cols>=4:
        if atac_cols>=6:
    	    atac_df = pd.read_csv(atac_peaks_filename, header=None, comment='#', sep="\t", na_filter=False, usecols=[0, 1, 2,3,4,5],names=["atac.chrom","atac_peak.start","atac_peak.end", "atac_peak.name", "atac_peak.strand","atac_peak.score"])
        else:
            atac_df = pd.read_csv(atac_peaks_filename, header=None, comment='#', sep="\t", na_filter=False, usecols=[0, 1, 2,3],names=["atac.chrom","atac_peak.start","atac_peak.end", "atac_peak.name"])
            atac_df['atac_peak.strand']='.'
            atac_df['atac_peak.score']='.'
    if motif_cols==3:
        motif_df = pd.read_csv(tf_motif_filename, header=None, comment='#', sep="\t",usecols=[0, 1, 2],names=["motif_region.chrom","motif_region.start","motif_region.end"])
        motif_df['motif_region.name']='.'
        motif_df['motif_region.strand']='.'
        motif_df['motif_region.score']='.'
    if motif_cols>=4:
        if motif_cols>=6:
            motif_df = pd.read_csv(tf_motif_filename, header=None, comment='#', sep="\t",usecols=[0, 1, 2,3,4,5],names=["motif_region.chrom","motif_region.start","motif_region.end", "motif_region.name", "motif_region.strand","motif_region.score"])
        else:
            motif_df = pd.read_csv(tf_motif_filename, header=None, comment='#', sep="\t",usecols=[0, 1, 2,3],names=["motif_region.chrom","motif_region.start","motif_region.end", "motif_region.name"])
            motif_df['motif_region.strand']='.'
            motif_df['motif_region.score']='.'
    #step 2 merge them with the final dataframe
    new_df = pd.merge(finaldf, motif_df,  how='left', left_on=['motif_region.chrom','motif_region.start','motif_region.end'], right_on = ['motif_region.chrom','motif_region.start','motif_region.end'])
    new_df = pd.merge(new_df, atac_df,  how='left', left_on=["atac.chrom","atac_peak.start","atac_peak.end"], right_on = ["atac.chrom","atac_peak.start","atac_peak.end"])
    #step three, put the columns in the order I want
    out_df = new_df[["motif_region.chrom","motif_region.start","motif_region.end", "motif_region.name","atac.chrom","atac_peak.start","atac_peak.end", "atac_peak.name", "atac_peak.strand","atac_peak.score"]]
    print (out_df)
    return out_df
    
# this will be ran in parallel
def find_motifs_in_chrom(current_chrom, files,verboseoption,bedfilenamepre,removemotifscountedmorethanonce,removemotififmidisoutsideH):
    testmode=True
    tf_motif_filename, atac_peaks_filename = files
    H = 1500          # in bps, the MD-score parameter (large window)
    h = 150           # in bps, the MD-score parameter (small window)
    #rootTF = tf_motif_filename.split("/")[-1]
    REPRESSOR_MARGIN = 500      # in bps, distance from the large window boundaries
    atac_df = pd.read_csv(atac_peaks_filename, header=None, comment='#', sep="\t", usecols=[0, 1, 2], \
                          names=['chrom', 'start', 'end'], na_filter=False, dtype={'chrom':'str', 'start':'int', 'end':'int'})
    atac_iter = atac_df[(atac_df.chrom == current_chrom)].itertuples()
    motif_df = pd.read_csv(tf_motif_filename, header=None, comment='#', sep="\t", usecols=[0, 1, 2], \
                           names=['chrom', 'start', 'end'], na_filter=False, dtype={'chrom':'str', 'start':'int', 'end':'int'})
    if len(motif_df) == 0:
        return None
    motif_iter = motif_df[(motif_df.chrom == current_chrom)].itertuples()
    last_motif = None
    total_motif_sites = 0
    keepoverlaps = []
    try:
        motif_region = next(motif_iter)   # get the first motif in the list
        total_motif_sites += 1
    except StopIteration:
        print('No motifs for chromosome ' + current_chrom + ' on file ' + tf_motif_filename)
        return None

    peak_count_overlapping_motif = 0
    for atac_peak in atac_iter:
        motifs_within_region = True

        atac_median = atac_peak.start + (atac_peak.end - atac_peak.start)/2
        # check the last motif, too, in case any of them falls within the region of
        # interest of two sequential ATAC-Seq peaks
        if last_motif:
            # account for those within the larger window (H)
            if is_in_window(last_motif, atac_median, H):
                line = [current_chrom, motif_region.start, motif_region.end, current_chrom, atac_peak.start, atac_peak.end, "added by if to g_H"]
                keepoverlaps.append(line)
                try:
                    motif_region = next(motif_iter)   # get the next motif in the list
                    total_motif_sites += 1
                except StopIteration:
                    pass
                last_motif = motif_region
                if motif_region.start > (atac_median + H):
                    motifs_within_region = False

        # Move to the next putative motif sites until we get one past our evaluation window
        while motifs_within_region:
            # account for those within the larger window (H)
            if is_in_window(motif_region, atac_median, H):
                line = [current_chrom, motif_region.start, motif_region.end, current_chrom, atac_peak.start, atac_peak.end, "added by while to g_H"]
                keepoverlaps.append(line)
            # if we still haven't shifted past this peak...
            if motif_region.start <= (atac_median + H):
                try:
                    motif_region = next(motif_iter)   # get the next motif in the list
                    total_motif_sites += 1
                except StopIteration:
                    # No more TF motifs for this chromosome
                    break
                last_motif = motif_region
            else:
                motifs_within_region = False
    # Count any remaining TF motif sites after the last ATAC peak
    while(len(motif_region) > 0):
        try:
            motif_region = next(motif_iter)   # this gets the next motif in the list
            total_motif_sites += 1
        except StopIteration:
            break
    df = pd.DataFrame.from_records(keepoverlaps, columns=["motif_region.chrom", "motif_region.start", "motif_region.end", "atac.chrom", "atac_peak.start", "atac_peak.end", "added by"])
    if verboseoption=="True":
        filename = bedfilenamepre+"__"+current_chrom+"_all.bed"
        df.to_csv(filename)
    df = df.drop(['added by'], axis=1)
    dupdf = df[df.duplicated()]
    if verboseoption=="True":
        if testmode:
            filename = bedfilenamepre+"__"+current_chrom+".bed"
            df.to_csv(filename) 
    dupdfrows, dupdfcolumns = dupdf.shape
    df = df.drop_duplicates()
    if verboseoption=="True":
          if testmode:
              filename = bedfilenamepre+"__"+current_chrom+"_dropdups.bed"
              df.to_csv(filename)
              if dupdfrows>0:
                  filename = bedfilenamepre+"__"+current_chrom+"_dups.bed"
                  dupdf.to_csv(filename)
          else:
             filename = bedfilenamepre+"__"+current_chrom+".bed"
             df.to_csv(filename)
    allmotifstarts = (df["motif_region.start"].tolist())
    df["atac_median"] = df["atac_peak.start"] + (df["atac_peak.end"] - df["atac_peak.start"])/2
    df["motif_median"] = df["motif_region.start"] + (df["motif_region.end"] - df["motif_region.start"])/2
    df["tf_distance"] = df["atac_median"]-df["motif_median"]
    dfrows, dfcolumns = df.shape
    removemotifscountedmorethanonce=True #this is there in case people only want the motif assigend to one ATAC peak
    removemotififmidisoutsideH=True #this allows people to remove motifs that are to close to the edge of H to be drawn in the barcode
    if removemotifscountedmorethanonce:
        problems = list(set([i for i in allmotifstarts if allmotifstarts.count(i)>1]))
        if len(problems)>0:
            df["keep"]=df.apply(lambda row: markeachproblem(df, row["motif_region.start"], row["atac_peak.start"], problems), axis=1)
            dfrows_filterproblemmotifs, dfcolumns_filterproblemmotifs = df.shape
            if verboseoption=="True":
                if testmode:
                   filename = bedfilenamepre+"__"+current_chrom+"_removerepeatmotifs.bed"
                   df.to_csv(filename)
                else:
                   filename = bedfilenamepre+"__"+current_chrom+".bed"
                   df.to_csv(filename)
            df = df[df["keep"]=="Y"]
            df = df.drop(['keep'], axis=1)
    if removemotififmidisoutsideH:
        g_Hdf = df[df['tf_distance']<=H]
        g_Hdfrows, g_Hdfcolumns = g_Hdf.shape
        if verboseoption=="True":
            if testmode:
                filename = bedfilenamepre+"__"+current_chrom+"_withinH.bed"
                g_Hdf.to_csv(filename)
            else:
                filename = bedfilenamepre+"__"+current_chrom+".bed"
                g_Hdf.to_csv(filename)
        g_hdf = g_Hdf[(g_Hdf['tf_distance']<=h) & (g_Hdf['tf_distance']>=-h)]
        g_hdfrows, g_hdfcolumns = g_hdf.shape
        tf_distances = g_Hdf["tf_distance"].tolist()
    else:
        g_Hdfrows, g_Hdfcolumns = df.shape
        g_hdf = df[(df['tf_distance']<=h) & (df['tf_distance']>=-h)]
        g_hdfrows, g_hdfcolumns = g_hdf.shape
        tf_distances = df["tf_distance"].tolist()
    return [tf_distances, g_hdfrows, g_Hdfrows, total_motif_sites]


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def markeachproblem(df, motifstart, atacstart, problems):
    if motifstart in problems:
        thismotifdf = df[df["motif_region.start"]==motifstart]
        mindis = df["tf_distance"].min()
        correctatacstartdf = thismotifdf[thismotifdf["tf_distance"]==thismotifdf["tf_distance"].min()]
        correctatacstart = correctatacstartdf["atac_peak.start"].tolist()[0]#atac_peak.start
        if atacstart==correctatacstart:
            return "Y"
        else:
            return "N"
    else:
        return "Y"

def get_md_score(tf_motif_filename, mp_threads, atac_peaks_filename, genome,verboseoption,output_dir,removemotifscountedmorethanonce,removemotififmidisoutsideH):
    #Get chromosomes for mutliprocessing
    chr_size_file = pybedtools.chromsizes(genome)
    unique_chr = list(chr_size_file.keys())[0:]
    CHROMOSOMES = [word for word in unique_chr if len(word) <= 6]
    if verboseoption=="True":
        rootTF = os.path.splitext(os.path.basename(tf_motif_filename))[0]+"_"
        output_prefix = os.path.splitext(os.path.basename(atac_peaks_filename))[0]
        boutput_dir = os.path.join(output_dir, "temp")
        boutput_dir = os.path.join(boutput_dir, "")
        ensure_dir(boutput_dir)
        bedfilenamepre = boutput_dir+rootTF+output_prefix
        finalbeddir = os.path.join(output_dir, "bedfiles")
        finalbeddir = os.path.join(finalbeddir, "")
        ensure_dir(finalbeddir)
        finalbedfile = finalbeddir+rootTF+output_prefix+".bed"
    HISTOGRAM_BINS = 150
    pool = multiprocessing.Pool(mp_threads)
    results = pool.map(partial( find_motifs_in_chrom, \
                                files=[tf_motif_filename, atac_peaks_filename],verboseoption=verboseoption,bedfilenamepre=bedfilenamepre,removemotifscountedmorethanonce=removemotifscountedmorethanonce,removemotififmidisoutsideH=removemotififmidisoutsideH), \
                       CHROMOSOMES)
    pool.close()
    pool.join()
    results_matching_motif = [x for x in results if x is not None]
    if len(results_matching_motif) > 0:
        sums = np.sum(results_matching_motif, axis=0)
        overall_g_h = sums[1]
        overall_g_H = sums[2]
        overall_motif_sites = sums[3]

        # Calculate the heatmap for this motif's barcode
        #tf_distances = reduce(lambda a, b: [*a, *b], [x[0] for x in results if x is not None])
        tf_distances = sums[0]
        heatmap, xedges = np.histogram(tf_distances, bins=HISTOGRAM_BINS)
        str_heatmap = np.char.mod('%d', heatmap)
        # TODO: Use the motif sequences to generate a logo for each motif, based
        #       only on the overlapping ATAC-Seq peaks
        if overall_g_H >= 0:
            if verboseoption=="True":
            #suck in a bed files and spit out a sorted bed file then delete all the temp files
                dfs = []
                for chrname in CHROMOSOMES:
                   bedfile = bedfilenamepre+"__"+chrname+".bed"
                   try:
                       df = pd.read_csv(bedfile)
                       dfs.append(df)
                   except:
                       pass
                finaldf = pd.concat(dfs,ignore_index=True,sort=False)       
                #output as a bed file and combine input files to get bed12
                finaldf = makebed12(finaldf, tf_motif_filename, atac_peaks_filename)
                finaldf.to_csv(finalbedfile,sep="\t",index=False)
            return [float(overall_g_h + 1)/(overall_g_H + 1), \
                    (overall_g_h + 1), \
                    (overall_g_H + 1), \
                    (overall_motif_sites + 1), \
                    ';'.join(str_heatmap)]
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='This script analyzes ATAC-Seq and GRO-Seq data and produces various plots for further data analysis.', epilog='IMPORTANT: Please ensure that ALL bed files used with this script are sorted by the same criteria.')
    parser.add_argument('-e', '--atac-peaks', dest='atac_peaks_filename', \
                        help='Full path to the ATAC-Seq broadPeak file.', \
                        default='', required=True)
    parser.add_argument('-m', '--motif-path', dest='tf_motif_path', \
                        help='Path to the location of the motif sites for the desired reference genome (i.e., "/usr/local/motifs/human/hg19/*").', \
                        default='', required=True)
    parser.add_argument('-g', '--genome', dest='genome', \
                        help='Genome to which the organism is mapped (e.g. hg38, mm10)', \
                        default='hg38', required=True)
    parser.add_argument('-t', '--threads', dest='mp_threads', \
                        help='Number of CPUs to use for multiprocessing of MD-score calculations. Depends on your hardware architecture.', \
                        default='1', required=False)
    parser.add_argument('-o', '--output', dest='output_dir', \
                        help='Path to where scores file will be saved. Save output will be your peak file rootname + _md_scores.txt.', \
                        default='', required=True)
    parser.add_argument('--verbose',action='store_true',\
                        help='This will output files in the outdirectory that contain the overlap of ATAC peaks and motifs for each chromosome.')
    args = parser.parse_args()
    verboseoption="False"
    if args.verbose:
        verboseoption="True"
        print ("verbose is on")
    #evaluation_radius = 750   # in bps
    ATAC_WIDTH_THRESHOLD = 5000   # in bp
    removemotifscountedmorethanonce=True #this is here in case people only want the motif assigend to one ATAC peak
    removemotififmidisoutsideH=True #this allows people to remove motifs whose midpoints are outside of H
    print('Starting --- ' + str(datetime.datetime.now()))
    atac_peaks_file = open(args.atac_peaks_filename)
    output_prefix = os.path.splitext(os.path.basename(args.atac_peaks_filename))[0]
    atac_csv_reader = csv.reader(atac_peaks_file, delimiter='\t')
    atac_line = next(atac_csv_reader)
    atac_peak_count = 0
    # skip the BedGraph headers
    while(atac_line[0][0] == '#'):
        atac_line = next(atac_csv_reader)

    # Determine the evaluation radius from the data itself
    all_widths = []
    while(atac_line):  # count the rest of ATAC peaks
        try:
            new_peak_width = int(atac_line[2]) - int(atac_line[1])
            if new_peak_width < ATAC_WIDTH_THRESHOLD:
                all_widths.append(new_peak_width)
            atac_line = next(atac_csv_reader)
        except StopIteration:
            break
        except IndexError:
            print("\nThere was an error with the ATAC-seq peaks file.\nPlease verify it conforms with a BedGraph-like format\n(tab-separated columns, any other lines commented with a leading '#')")
            sys.exit(1)
    atac_peak_mean = np.mean(all_widths)
    atac_peak_std = np.std(all_widths)
    evaluation_radius = (int(atac_peak_mean) + 2 * int(atac_peak_std)) / 2
    print('ATAC mean width: %d bp (std: %d bp). Determined an evaluation radius of %d bp' % \
          (atac_peak_mean, atac_peak_std, evaluation_radius))

    # Start reading from the top of the ATAC-Seq BedGraph again
    atac_peaks_file.seek(0)
    atac_csv_reader = csv.reader(atac_peaks_file, delimiter='\t')
    atac_line = next(atac_csv_reader)
    while(atac_line[0][0] == '#'):
        atac_line = next(atac_csv_reader)

    motif_stats = []
    sorted_motif_stats = []
    tf_motif_path = args.tf_motif_path + '/*'
    motif_filenames = glob.glob(tf_motif_path)
    motif_count = len(motif_filenames)
    print("Processing motif files in %s" % tf_motif_path)
    for filename in motif_filenames:
        filename_no_path = filename.split('/')[-1]
        if os.path.getsize(filename) > 0 and \
           os.path.basename(filename).endswith(tuple(['.bed', '.BedGraph', '.txt'])):
            [md_score, small_window, large_window, motif_site_count, heat] = get_md_score(filename, int(args.mp_threads), args.atac_peaks_filename, args.genome,verboseoption, args.output_dir, removemotifscountedmorethanonce,removemotififmidisoutsideH)
            print('The MD-score for ATAC reads vs %s is %.6f' % (filename_no_path, md_score))
            motif_stats.append({ 'motif_file': filename_no_path, \
                                 'md_score': md_score, \
                                 'small_window': small_window, \
                                 'large_window': large_window, \
                                 'motif_site_count': motif_site_count, \
                                 'heat': heat })

    # sort the stats dictionary by MD-score, descending order
    sorted_motif_stats = sorted(motif_stats, key=itemgetter('md_score'), reverse=True)

    md_score_fp = open("%s/%s_md_scores.txt" % (args.output_dir, output_prefix), 'w')
    for stat in sorted_motif_stats:
        md_score_fp.write("%s,%s,%s,%s,%s,%s\n" % \
                          (stat['motif_file'], stat['md_score'], stat['small_window'], \
                           stat['large_window'], stat['motif_site_count'], stat['heat']))
    md_score_fp.close()
    print('All done --- %s' % str(datetime.datetime.now()))
    sys.exit(0)


if __name__=='__main__':
    main()
