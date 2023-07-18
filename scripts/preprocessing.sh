# %%capture
#@title  { form-width: "50%" }
#@markdown Run __ONCE__ to initialize application


# DOWNLOADING JEDAI JAR FILES

curl --remote-name \
     -H 'Accept: application/vnd.github.v3.raw' \
     --location https://raw.githubusercontent.com/AI-team-UoA/pyJedAI/main/tests/jedai-core.jar

curl --remote-name \
     -H 'Accept: application/vnd.github.v3.raw' \
     --location https://raw.githubusercontent.com/AI-team-UoA/pyJedAI/main/tests/jedai-core-with-joins.jar


pip install plotly


# PYJNIUS DOWNLOAD AND CONF

pip install pyjnius
java -version

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
mkdir -p /usr/lib/jvm/java-1.11.0-openjdk-amd64/jre/lib/amd64/server/
ln -s /usr/lib/jvm/java-1.11.0-openjdk-amd64/lib/server/libjvm.so /usr/lib/jvm/java-1.11.0-openjdk-amd64/jre/lib/amd64/server/libjvm.so

curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/dbpediaProfiles.rar
pip install unrar
unrar x "dbpediaProfiles.rar" "./"
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/dblpProfiles2
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/restaurantsIdDuplicates
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/abtBuyIdDuplicates
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/amazonGpIdDuplicates
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/dblpAcmIdDuplicates
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/imdbTmdbIdDuplicates
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/imdbTvdbIdDuplicates
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/tmdbTvdbIdDuplicates
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/amazonWalmartIdDuplicates
curl --remote-name \
  -H 'Accept: application/vnd.github.v3.raw' \
  --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/dblpScholarIdDuplicates
curl --remote-name \
  -H 'Accept: application/vnd.github.v3.raw' \
  --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/moviesIdDuplicates
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/restaurant1Profiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/abtProfiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/amazonProfiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/dblpProfiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/tmdbProfiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/imdbProfilesNEW
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/imdbProfiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/walmartProfiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/restaurant2Profiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/buyProfiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/gpProfiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/acmProfiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/tmdbProfiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/tvdbProfiles
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/amazonProfiles2
curl --remote-name \
    -H 'Accept: application/vnd.github.v3.raw' \
    --location https://raw.githubusercontent.com/scify/JedAIToolkit/master/data/cleanCleanErDatasets/scholarProfiles

# PYJEDAI DOWNLOAD AND DATASETS

pip install pyjedai==0.0.6

wget https://zenodo.org/record/7460624/files/D2abt.csv
wget https://zenodo.org/record/7460624/files/D2buy.csv
wget https://zenodo.org/record/7460624/files/D2gt.csv

wget https://zenodo.org/record/7460624/files/D3amazon.csv
wget https://zenodo.org/record/7460624/files/D3gp.csv
wget https://zenodo.org/record/7460624/files/D3gt.csv

wget https://zenodo.org/record/7460624/files/D8amazon.csv
wget https://zenodo.org/record/7460624/files/D8gt.csv
wget https://zenodo.org/record/7460624/files/D8walmart.csv
