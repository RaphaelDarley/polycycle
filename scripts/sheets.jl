import XLSX
import DataFrames
import CSV
using Plots

# println("test");

xf = XLSX.readxlsx("polycycle_data.xlsx")

oil_sheet = xf["US oil prod"]

oil_frame = XLSX.eachtablerow(oil_sheet) |> DataFrames.DataFrame

vscodedisplay(oil_frame)

plot(oil_frame[!, "year"], oil_frame[!, "oil_prod"])


# CSV.write("oil_prod.csv", oil_frame)
