# Tool zur Durchführung der Experimente von der Masterarbeit "Conformal-Prediction zur Verbesserung der Vorhersage nicht-funktionaler Eigenschaften in konfigurierbaren Softwaresystemen" 

## Vorbereitung
- Installieren der Requirements
- Erstellen eines Ordners mit FeatureModel.xml wie für SPLConceror (https://github.com/se-sic/SPLConqueror) benötigt
- Einfügen der measurements.csv mit den Feature-Spalten und gemessener nicht-funktionaler Eigenschaft
- Einfügen der config.json nach gegebenen Muster

## Ausführung
- Ausführen von experiment.py mit Parameter: Pfad des oben erstellten Ordners mit FeatureModel.xml, measurements.csv und config.json
- Falls einzelne Runs keine Ergebnisse liefern können diese mit ausführen von fix_experiment mit Parameter: Ordnerpfad

## Ergebnisse
- Es werden innerhalb des Ordners unterordner erstellt mit dem Namen der Sampling-Strategie wie in config.json
- Innerhalb dieser Ordner sind die einzelnen Runs als Unterordner zu finden
- Innerhalb der Runs findet man das Trainingsset (train.csv), das Kalibrierungsset (calib.csv) und das Validierungsset (test.csv)
- Außerdem findet man für jede gewählte Methode 3 Ergebnis-Dateien die jeweils mit Conformal-Prediction-Methode, Modell und Non-conformity-Measure namentlich anfangen. Es gibt eine .csv mit den Punkt und Intervall Vorhersagen, eine .txt mit den errechneten Metriken und ein .png mit einem entsprechendem Plot
- Für weiteres Explorieren gibt es ein Repository mit Jupyter-Notebooks (https://github.com/Stefan577/Masterthesis-Explore)