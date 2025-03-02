# DropBoxの共有リンクを設定
$url = "https://www.dropbox.com/scl/fi/adhk8j33mb1hnpx36i6x3/Scene2.zip?rlkey=yj7k1a6q02zq1zhjij6ifn8iu&st=qvj82evt&dl=1"

# ダウンロード先のパス
$outputPath = "Scene2.zip"
# 解凍先の一時フォルダ
$tempExtractPath = "extracted_files"
# 最終的な解凍先のフォルダ
$finalExtractPath = "Scene2"

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
    $extractedFolder = Join-Path -Path $tempExtractPath -ChildPath "Scene2"
    if (Test-Path -Path $extractedFolder) {
        Write-Output "Renaming extracted directory..."
        Rename-Item -Path $extractedFolder -NewName (Split-Path -Path $finalExtractPath -Leaf)
        
        # 解凍先の最終フォルダに移動
        Move-Item -Path (Join-Path -Path $tempExtractPath -ChildPath "Scene2") -Destination $finalExtractPath
    }

    # 一時フォルダを削除（オプション）
    Remove-Item -Path $tempExtractPath -Recurse -Force
    Remove-Item -Path $outputPath
} else {
    Write-Output "The final extraction folder already exists. Skipping download and extraction."
}
