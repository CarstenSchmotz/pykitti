# Öffne die Binärdatei im Lesemodus ('rb' für Binärlesen)
with open('D:\\Dokumente\\01_BA_Git\\pykitti\\kitty\\2011_09_26\\2011_09_26_drive_0001_sync\\velodyne_points\\data\\0000000000.bin', 'rb') as f:

    # Lese den gesamten Inhalt der Datei
    content = f.read()

# Gib den Inhalt der Datei aus
print(content)
