# chromab_corrector
A semi-automatic image enhancing software specialized in reducing the chromatic aberrations for astrophotography
<h2>GUI Version</h2>
Requires a JRE. Unzip and run chromab.jar
<h2>CLI arguments</h2>
<ol>
<li><h4>file</h4>
the absolute path of the image to process (path must exclude white spaces)
</li>
<li><h4>max_distance</h4>
<i>integer</i><br/>
Intercorrelation distance (cf doc) <br/>
<strong>default : 10px</strong>
</li>
<li><h4>origin_max_distance</h4>
<i>integer</i><br/>
Maximum distance to computed origin for more precise barycenter computation<br/>
<strong>default : 100px</strong></li>
<li><h4>max_intersect_distance</h4>
<i>integer</i><br/>
Maximum distance for scattering discrimination<br/>
<strong>default : 10% </strong></li>
<li><h4>max_dot_product</h4>
<i>float</i><br/>
Max dot product between U = (origin to enlargement vector origin) and V = enlargement vector<br/>
<strong>default : 0.05 </strong></li>
</ol>

<h2>Example</h2>
<code> python chromab.py --file C:/Users/User/img.png --max_distance 10 --origin_max_distance 100 --max_intersect_distance 10 --max_dot_product 0.05</code>

<h3>NB</h3>
If not specified, the algorithm will use default values.

[Buy me a coffee](https://www.buymeacoffee.com/asnarok)
