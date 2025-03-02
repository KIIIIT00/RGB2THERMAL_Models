# DropBoxの共有リンクを設定
$url = "https://www.dropbox.com/scl/fi/s01b3uxrt4c78sgg9e7v3/Scene2ver2.zip?rlkey=v9ptdeuev0blxkcz6sf9lwggm&st=7c7bdsaz&dl=1"

# ダウンロード先のパス
$outputPath = "Scene2ver2.zip"
# 解凍先の一時フォルダ
$tempExtractPath = "extracted_files"
# 最終的な解凍先のフォルダ
$finalExtractPath = "Scene2ver2"

# 解凍先の一時フォルダが存在しない場合、作成
if (-Not (Test-Path -Path $tempExtractPath)) {
    New-Item -ItemType Directory -Path $tempExtractPath
}

# 解凍先のフォルダが存在するかチェック
if (-Not (Test-Path -Path $finalExtractPath)) {
    # フォルダが存在しない場合、ZIPファイルをダウンロード
    Write-Output "Downloading ZIP file..."
    Invoke-WebRequest -Uri $url -OutFile $outputPath
    
    # ZIPファイルを解凍
    Write-Output "Extracting ZIP file..."
    Expand-Archive -Path $outputPath -DestinationPath $tempExtractPath
    Write-Output "Extraction complete."

    # 解凍後のディレクトリ構造を変更
    $extractedFolder = Join-Path -Path $tempExtractPath -ChildPath "Scene2ver2"
if ((Split-Path -Leaf $extractedFolder) -ne (Split-Path -Leaf $finalExtractPath)) {
        Write-Output "Renaming extracted directory..."
        Rename-Item -Path $extractedFolder -NewName (Split-Path -Path $finalExtractPath -Leaf)
    }
        
        # 解凍先の最終フォルダに移動
        Move-Item -Path (Join-Path -Path $tempExtractPath -ChildPath "Scene2ver2") -Destination $finalExtractPath

    # 一時フォルダを削除（オプション）
    Remove-Item -Path $tempExtractPath -Recurse -Force
    Remove-Item -Path $outputPath
} else {
    Write-Output "The final extraction folder already exists. Skipping download and extraction."
}
