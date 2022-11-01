
# import jnius_config;
# jnius_config.add_classpath('jedai-core-3.2.1-jar-with-dependencies.jar')
# from jnius import autoclass

# BasicConfigurator =  autoclass('org.apache.log4j.BasicConfigurator')
# IBlockBuilding =  autoclass('org.scify.jedai.blockbuilding.IBlockBuilding')
# StandardBlocking = autoclass('org.scify.jedai.blockbuilding.StandardBlocking')
# CardinalityEdgePruning = autoclass('org.scify.jedai.blockprocessing.comparisoncleaning.CardinalityEdgePruning')
# GtCSVReader = autoclass('org.scify.jedai.datareader.groundtruthreader.GtCSVReader')
# BlocksPerformanceWriter = autoclass('org.scify.jedai.datawriter.BlocksPerformanceWriter')
# BlocksPerformance = autoclass('org.scify.jedai.utilities.BlocksPerformance')
# AbstractDuplicatePropagation = autoclass('org.scify.jedai.utilities.datastructures.AbstractDuplicatePropagation')
# UnilateralDuplicatePropagation = autoclass('org.scify.jedai.utilities.datastructures.UnilateralDuplicatePropagation')
# BilateralDuplicatePropagation = autoclass('org.scify.jedai.utilities.datastructures.BilateralDuplicatePropagation')
# EntityCSVReader = autoclass('org.scify.jedai.datareader.entityreader.EntityCSVReader')
# StringBuilder  = autoclass('java.lang.StringBuilder')
# File = autoclass('java.io.File')
# PrintWriter = autoclass('java.io.PrintWriter')
# List = autoclass('java.util.List')
# Set = autoclass('java.util.Set')

# X1 = "../data/ccer/D2/abt.csv"
# Y1 = "../data/ccer/D2/buy.csv"
# GT = "../data/ccer/D2/gt.csv"

# csvReader = EntityCSVReader(X1)
# csvReader.setAttributeNamesInFirstRow(True)
# csvReader.setSeparator('|')
# csvReader.setIdIndex(0)

# profiles1 = csvReader.getEntityProfiles()
# print("Entities from Dataset X1: " + str(profiles1.size()))

# csvReader = EntityCSVReader(Y1)
# csvReader.setAttributeNamesInFirstRow(True)
# csvReader.setSeparator('|')
# csvReader.setIdIndex(0)

# profiles2 = csvReader.getEntityProfiles()
# print("Entities from Dataset X2: " + str(profiles2.size()))

# gtCsvReader = GtCSVReader(GT)
# gtCsvReader.setIgnoreFirstRow(True)
# gtCsvReader.setSeparator("|")
# duplicates = gtCsvReader.getDuplicatePairs(profiles1, profiles2)
# duplicatePropagation = BilateralDuplicatePropagation(duplicates)
# print("Duplicates\t:\t" + str(duplicates.size()))

# # blockBuildingMethod = StandardBlocking()
# # blocks = blockBuildingMethod.getBlocks(profiles)