---
layout: post
title: Advanced Regression Housing Prediction
---


<html>
<head><meta charset="utf-8">
<title>Advance regression housing prediction modified data files</title>


<style type="text/css">
    /<em>!
</em>
<em> Twitter Bootstrap
</em>
<em>/
/</em>!
 <em> Bootstrap v3.3.7 (<a href="http://getbootstrap.com">http://getbootstrap.com</a>)
 </em> Copyright 2011-2016 Twitter, Inc.
 <em> Licensed under MIT (<a href="https://github.com/twbs/bootstrap/blob/master/LICENSE">https://github.com/twbs/bootstrap/blob/master/LICENSE</a>)
 </em>/
/<em>! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css </em>/
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type=&quot;button&quot;],
input[type=&quot;reset&quot;],
input[type=&quot;submit&quot;] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type=&quot;checkbox&quot;],
input[type=&quot;radio&quot;] {
  box-sizing: border-box;
  padding: 0;
}
input[type=&quot;number&quot;]::-webkit-inner-spin-button,
input[type=&quot;number&quot;]::-webkit-outer-spin-button {
  height: auto;
}
input[type=&quot;search&quot;] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type=&quot;search&quot;]::-webkit-search-cancel-button,
input[type=&quot;search&quot;]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/<em>! Source: <a href="https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css">https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css</a> </em>/
@media print {
  <em>,
  </em>:before,
  <em>:after {
    background: transparent !important;
    color: #000 !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: &quot; (&quot; attr(href) &quot;)&quot;;
  }
  abbr[title]:after {
    content: &quot; (&quot; attr(title) &quot;)&quot;;
  }
  a[href^=&quot;#&quot;]:after,
  a[href^=&quot;javascript:&quot;]:after {
    content: &quot;&quot;;
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn &gt; .caret,
  .dropup &gt; .btn &gt; .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: &#39;Glyphicons Halflings&#39;;
  src: url(&#39;../components/bootstrap/fonts/glyphicons-halflings-regular.eot&#39;);
  src: url(&#39;../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix&#39;) format(&#39;embedded-opentype&#39;), url(&#39;../components/bootstrap/fonts/glyphicons-halflings-regular.woff2&#39;) format(&#39;woff2&#39;), url(&#39;../components/bootstrap/fonts/glyphicons-halflings-regular.woff&#39;) format(&#39;woff&#39;), url(&#39;../components/bootstrap/fonts/glyphicons-halflings-regular.ttf&#39;) format(&#39;truetype&#39;), url(&#39;../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular&#39;) format(&#39;svg&#39;);
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: &#39;Glyphicons Halflings&#39;;
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: &quot;\002a&quot;;
}
.glyphicon-plus:before {
  content: &quot;\002b&quot;;
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: &quot;\20ac&quot;;
}
.glyphicon-minus:before {
  content: &quot;\2212&quot;;
}
.glyphicon-cloud:before {
  content: &quot;\2601&quot;;
}
.glyphicon-envelope:before {
  content: &quot;\2709&quot;;
}
.glyphicon-pencil:before {
  content: &quot;\270f&quot;;
}
.glyphicon-glass:before {
  content: &quot;\e001&quot;;
}
.glyphicon-music:before {
  content: &quot;\e002&quot;;
}
.glyphicon-search:before {
  content: &quot;\e003&quot;;
}
.glyphicon-heart:before {
  content: &quot;\e005&quot;;
}
.glyphicon-star:before {
  content: &quot;\e006&quot;;
}
.glyphicon-star-empty:before {
  content: &quot;\e007&quot;;
}
.glyphicon-user:before {
  content: &quot;\e008&quot;;
}
.glyphicon-film:before {
  content: &quot;\e009&quot;;
}
.glyphicon-th-large:before {
  content: &quot;\e010&quot;;
}
.glyphicon-th:before {
  content: &quot;\e011&quot;;
}
.glyphicon-th-list:before {
  content: &quot;\e012&quot;;
}
.glyphicon-ok:before {
  content: &quot;\e013&quot;;
}
.glyphicon-remove:before {
  content: &quot;\e014&quot;;
}
.glyphicon-zoom-in:before {
  content: &quot;\e015&quot;;
}
.glyphicon-zoom-out:before {
  content: &quot;\e016&quot;;
}
.glyphicon-off:before {
  content: &quot;\e017&quot;;
}
.glyphicon-signal:before {
  content: &quot;\e018&quot;;
}
.glyphicon-cog:before {
  content: &quot;\e019&quot;;
}
.glyphicon-trash:before {
  content: &quot;\e020&quot;;
}
.glyphicon-home:before {
  content: &quot;\e021&quot;;
}
.glyphicon-file:before {
  content: &quot;\e022&quot;;
}
.glyphicon-time:before {
  content: &quot;\e023&quot;;
}
.glyphicon-road:before {
  content: &quot;\e024&quot;;
}
.glyphicon-download-alt:before {
  content: &quot;\e025&quot;;
}
.glyphicon-download:before {
  content: &quot;\e026&quot;;
}
.glyphicon-upload:before {
  content: &quot;\e027&quot;;
}
.glyphicon-inbox:before {
  content: &quot;\e028&quot;;
}
.glyphicon-play-circle:before {
  content: &quot;\e029&quot;;
}
.glyphicon-repeat:before {
  content: &quot;\e030&quot;;
}
.glyphicon-refresh:before {
  content: &quot;\e031&quot;;
}
.glyphicon-list-alt:before {
  content: &quot;\e032&quot;;
}
.glyphicon-lock:before {
  content: &quot;\e033&quot;;
}
.glyphicon-flag:before {
  content: &quot;\e034&quot;;
}
.glyphicon-headphones:before {
  content: &quot;\e035&quot;;
}
.glyphicon-volume-off:before {
  content: &quot;\e036&quot;;
}
.glyphicon-volume-down:before {
  content: &quot;\e037&quot;;
}
.glyphicon-volume-up:before {
  content: &quot;\e038&quot;;
}
.glyphicon-qrcode:before {
  content: &quot;\e039&quot;;
}
.glyphicon-barcode:before {
  content: &quot;\e040&quot;;
}
.glyphicon-tag:before {
  content: &quot;\e041&quot;;
}
.glyphicon-tags:before {
  content: &quot;\e042&quot;;
}
.glyphicon-book:before {
  content: &quot;\e043&quot;;
}
.glyphicon-bookmark:before {
  content: &quot;\e044&quot;;
}
.glyphicon-print:before {
  content: &quot;\e045&quot;;
}
.glyphicon-camera:before {
  content: &quot;\e046&quot;;
}
.glyphicon-font:before {
  content: &quot;\e047&quot;;
}
.glyphicon-bold:before {
  content: &quot;\e048&quot;;
}
.glyphicon-italic:before {
  content: &quot;\e049&quot;;
}
.glyphicon-text-height:before {
  content: &quot;\e050&quot;;
}
.glyphicon-text-width:before {
  content: &quot;\e051&quot;;
}
.glyphicon-align-left:before {
  content: &quot;\e052&quot;;
}
.glyphicon-align-center:before {
  content: &quot;\e053&quot;;
}
.glyphicon-align-right:before {
  content: &quot;\e054&quot;;
}
.glyphicon-align-justify:before {
  content: &quot;\e055&quot;;
}
.glyphicon-list:before {
  content: &quot;\e056&quot;;
}
.glyphicon-indent-left:before {
  content: &quot;\e057&quot;;
}
.glyphicon-indent-right:before {
  content: &quot;\e058&quot;;
}
.glyphicon-facetime-video:before {
  content: &quot;\e059&quot;;
}
.glyphicon-picture:before {
  content: &quot;\e060&quot;;
}
.glyphicon-map-marker:before {
  content: &quot;\e062&quot;;
}
.glyphicon-adjust:before {
  content: &quot;\e063&quot;;
}
.glyphicon-tint:before {
  content: &quot;\e064&quot;;
}
.glyphicon-edit:before {
  content: &quot;\e065&quot;;
}
.glyphicon-share:before {
  content: &quot;\e066&quot;;
}
.glyphicon-check:before {
  content: &quot;\e067&quot;;
}
.glyphicon-move:before {
  content: &quot;\e068&quot;;
}
.glyphicon-step-backward:before {
  content: &quot;\e069&quot;;
}
.glyphicon-fast-backward:before {
  content: &quot;\e070&quot;;
}
.glyphicon-backward:before {
  content: &quot;\e071&quot;;
}
.glyphicon-play:before {
  content: &quot;\e072&quot;;
}
.glyphicon-pause:before {
  content: &quot;\e073&quot;;
}
.glyphicon-stop:before {
  content: &quot;\e074&quot;;
}
.glyphicon-forward:before {
  content: &quot;\e075&quot;;
}
.glyphicon-fast-forward:before {
  content: &quot;\e076&quot;;
}
.glyphicon-step-forward:before {
  content: &quot;\e077&quot;;
}
.glyphicon-eject:before {
  content: &quot;\e078&quot;;
}
.glyphicon-chevron-left:before {
  content: &quot;\e079&quot;;
}
.glyphicon-chevron-right:before {
  content: &quot;\e080&quot;;
}
.glyphicon-plus-sign:before {
  content: &quot;\e081&quot;;
}
.glyphicon-minus-sign:before {
  content: &quot;\e082&quot;;
}
.glyphicon-remove-sign:before {
  content: &quot;\e083&quot;;
}
.glyphicon-ok-sign:before {
  content: &quot;\e084&quot;;
}
.glyphicon-question-sign:before {
  content: &quot;\e085&quot;;
}
.glyphicon-info-sign:before {
  content: &quot;\e086&quot;;
}
.glyphicon-screenshot:before {
  content: &quot;\e087&quot;;
}
.glyphicon-remove-circle:before {
  content: &quot;\e088&quot;;
}
.glyphicon-ok-circle:before {
  content: &quot;\e089&quot;;
}
.glyphicon-ban-circle:before {
  content: &quot;\e090&quot;;
}
.glyphicon-arrow-left:before {
  content: &quot;\e091&quot;;
}
.glyphicon-arrow-right:before {
  content: &quot;\e092&quot;;
}
.glyphicon-arrow-up:before {
  content: &quot;\e093&quot;;
}
.glyphicon-arrow-down:before {
  content: &quot;\e094&quot;;
}
.glyphicon-share-alt:before {
  content: &quot;\e095&quot;;
}
.glyphicon-resize-full:before {
  content: &quot;\e096&quot;;
}
.glyphicon-resize-small:before {
  content: &quot;\e097&quot;;
}
.glyphicon-exclamation-sign:before {
  content: &quot;\e101&quot;;
}
.glyphicon-gift:before {
  content: &quot;\e102&quot;;
}
.glyphicon-leaf:before {
  content: &quot;\e103&quot;;
}
.glyphicon-fire:before {
  content: &quot;\e104&quot;;
}
.glyphicon-eye-open:before {
  content: &quot;\e105&quot;;
}
.glyphicon-eye-close:before {
  content: &quot;\e106&quot;;
}
.glyphicon-warning-sign:before {
  content: &quot;\e107&quot;;
}
.glyphicon-plane:before {
  content: &quot;\e108&quot;;
}
.glyphicon-calendar:before {
  content: &quot;\e109&quot;;
}
.glyphicon-random:before {
  content: &quot;\e110&quot;;
}
.glyphicon-comment:before {
  content: &quot;\e111&quot;;
}
.glyphicon-magnet:before {
  content: &quot;\e112&quot;;
}
.glyphicon-chevron-up:before {
  content: &quot;\e113&quot;;
}
.glyphicon-chevron-down:before {
  content: &quot;\e114&quot;;
}
.glyphicon-retweet:before {
  content: &quot;\e115&quot;;
}
.glyphicon-shopping-cart:before {
  content: &quot;\e116&quot;;
}
.glyphicon-folder-close:before {
  content: &quot;\e117&quot;;
}
.glyphicon-folder-open:before {
  content: &quot;\e118&quot;;
}
.glyphicon-resize-vertical:before {
  content: &quot;\e119&quot;;
}
.glyphicon-resize-horizontal:before {
  content: &quot;\e120&quot;;
}
.glyphicon-hdd:before {
  content: &quot;\e121&quot;;
}
.glyphicon-bullhorn:before {
  content: &quot;\e122&quot;;
}
.glyphicon-bell:before {
  content: &quot;\e123&quot;;
}
.glyphicon-certificate:before {
  content: &quot;\e124&quot;;
}
.glyphicon-thumbs-up:before {
  content: &quot;\e125&quot;;
}
.glyphicon-thumbs-down:before {
  content: &quot;\e126&quot;;
}
.glyphicon-hand-right:before {
  content: &quot;\e127&quot;;
}
.glyphicon-hand-left:before {
  content: &quot;\e128&quot;;
}
.glyphicon-hand-up:before {
  content: &quot;\e129&quot;;
}
.glyphicon-hand-down:before {
  content: &quot;\e130&quot;;
}
.glyphicon-circle-arrow-right:before {
  content: &quot;\e131&quot;;
}
.glyphicon-circle-arrow-left:before {
  content: &quot;\e132&quot;;
}
.glyphicon-circle-arrow-up:before {
  content: &quot;\e133&quot;;
}
.glyphicon-circle-arrow-down:before {
  content: &quot;\e134&quot;;
}
.glyphicon-globe:before {
  content: &quot;\e135&quot;;
}
.glyphicon-wrench:before {
  content: &quot;\e136&quot;;
}
.glyphicon-tasks:before {
  content: &quot;\e137&quot;;
}
.glyphicon-filter:before {
  content: &quot;\e138&quot;;
}
.glyphicon-briefcase:before {
  content: &quot;\e139&quot;;
}
.glyphicon-fullscreen:before {
  content: &quot;\e140&quot;;
}
.glyphicon-dashboard:before {
  content: &quot;\e141&quot;;
}
.glyphicon-paperclip:before {
  content: &quot;\e142&quot;;
}
.glyphicon-heart-empty:before {
  content: &quot;\e143&quot;;
}
.glyphicon-link:before {
  content: &quot;\e144&quot;;
}
.glyphicon-phone:before {
  content: &quot;\e145&quot;;
}
.glyphicon-pushpin:before {
  content: &quot;\e146&quot;;
}
.glyphicon-usd:before {
  content: &quot;\e148&quot;;
}
.glyphicon-gbp:before {
  content: &quot;\e149&quot;;
}
.glyphicon-sort:before {
  content: &quot;\e150&quot;;
}
.glyphicon-sort-by-alphabet:before {
  content: &quot;\e151&quot;;
}
.glyphicon-sort-by-alphabet-alt:before {
  content: &quot;\e152&quot;;
}
.glyphicon-sort-by-order:before {
  content: &quot;\e153&quot;;
}
.glyphicon-sort-by-order-alt:before {
  content: &quot;\e154&quot;;
}
.glyphicon-sort-by-attributes:before {
  content: &quot;\e155&quot;;
}
.glyphicon-sort-by-attributes-alt:before {
  content: &quot;\e156&quot;;
}
.glyphicon-unchecked:before {
  content: &quot;\e157&quot;;
}
.glyphicon-expand:before {
  content: &quot;\e158&quot;;
}
.glyphicon-collapse-down:before {
  content: &quot;\e159&quot;;
}
.glyphicon-collapse-up:before {
  content: &quot;\e160&quot;;
}
.glyphicon-log-in:before {
  content: &quot;\e161&quot;;
}
.glyphicon-flash:before {
  content: &quot;\e162&quot;;
}
.glyphicon-log-out:before {
  content: &quot;\e163&quot;;
}
.glyphicon-new-window:before {
  content: &quot;\e164&quot;;
}
.glyphicon-record:before {
  content: &quot;\e165&quot;;
}
.glyphicon-save:before {
  content: &quot;\e166&quot;;
}
.glyphicon-open:before {
  content: &quot;\e167&quot;;
}
.glyphicon-saved:before {
  content: &quot;\e168&quot;;
}
.glyphicon-import:before {
  content: &quot;\e169&quot;;
}
.glyphicon-export:before {
  content: &quot;\e170&quot;;
}
.glyphicon-send:before {
  content: &quot;\e171&quot;;
}
.glyphicon-floppy-disk:before {
  content: &quot;\e172&quot;;
}
.glyphicon-floppy-saved:before {
  content: &quot;\e173&quot;;
}
.glyphicon-floppy-remove:before {
  content: &quot;\e174&quot;;
}
.glyphicon-floppy-save:before {
  content: &quot;\e175&quot;;
}
.glyphicon-floppy-open:before {
  content: &quot;\e176&quot;;
}
.glyphicon-credit-card:before {
  content: &quot;\e177&quot;;
}
.glyphicon-transfer:before {
  content: &quot;\e178&quot;;
}
.glyphicon-cutlery:before {
  content: &quot;\e179&quot;;
}
.glyphicon-header:before {
  content: &quot;\e180&quot;;
}
.glyphicon-compressed:before {
  content: &quot;\e181&quot;;
}
.glyphicon-earphone:before {
  content: &quot;\e182&quot;;
}
.glyphicon-phone-alt:before {
  content: &quot;\e183&quot;;
}
.glyphicon-tower:before {
  content: &quot;\e184&quot;;
}
.glyphicon-stats:before {
  content: &quot;\e185&quot;;
}
.glyphicon-sd-video:before {
  content: &quot;\e186&quot;;
}
.glyphicon-hd-video:before {
  content: &quot;\e187&quot;;
}
.glyphicon-subtitles:before {
  content: &quot;\e188&quot;;
}
.glyphicon-sound-stereo:before {
  content: &quot;\e189&quot;;
}
.glyphicon-sound-dolby:before {
  content: &quot;\e190&quot;;
}
.glyphicon-sound-5-1:before {
  content: &quot;\e191&quot;;
}
.glyphicon-sound-6-1:before {
  content: &quot;\e192&quot;;
}
.glyphicon-sound-7-1:before {
  content: &quot;\e193&quot;;
}
.glyphicon-copyright-mark:before {
  content: &quot;\e194&quot;;
}
.glyphicon-registration-mark:before {
  content: &quot;\e195&quot;;
}
.glyphicon-cloud-download:before {
  content: &quot;\e197&quot;;
}
.glyphicon-cloud-upload:before {
  content: &quot;\e198&quot;;
}
.glyphicon-tree-conifer:before {
  content: &quot;\e199&quot;;
}
.glyphicon-tree-deciduous:before {
  content: &quot;\e200&quot;;
}
.glyphicon-cd:before {
  content: &quot;\e201&quot;;
}
.glyphicon-save-file:before {
  content: &quot;\e202&quot;;
}
.glyphicon-open-file:before {
  content: &quot;\e203&quot;;
}
.glyphicon-level-up:before {
  content: &quot;\e204&quot;;
}
.glyphicon-copy:before {
  content: &quot;\e205&quot;;
}
.glyphicon-paste:before {
  content: &quot;\e206&quot;;
}
.glyphicon-alert:before {
  content: &quot;\e209&quot;;
}
.glyphicon-equalizer:before {
  content: &quot;\e210&quot;;
}
.glyphicon-king:before {
  content: &quot;\e211&quot;;
}
.glyphicon-queen:before {
  content: &quot;\e212&quot;;
}
.glyphicon-pawn:before {
  content: &quot;\e213&quot;;
}
.glyphicon-bishop:before {
  content: &quot;\e214&quot;;
}
.glyphicon-knight:before {
  content: &quot;\e215&quot;;
}
.glyphicon-baby-formula:before {
  content: &quot;\e216&quot;;
}
.glyphicon-tent:before {
  content: &quot;\26fa&quot;;
}
.glyphicon-blackboard:before {
  content: &quot;\e218&quot;;
}
.glyphicon-bed:before {
  content: &quot;\e219&quot;;
}
.glyphicon-apple:before {
  content: &quot;\f8ff&quot;;
}
.glyphicon-erase:before {
  content: &quot;\e221&quot;;
}
.glyphicon-hourglass:before {
  content: &quot;\231b&quot;;
}
.glyphicon-lamp:before {
  content: &quot;\e223&quot;;
}
.glyphicon-duplicate:before {
  content: &quot;\e224&quot;;
}
.glyphicon-piggy-bank:before {
  content: &quot;\e225&quot;;
}
.glyphicon-scissors:before {
  content: &quot;\e226&quot;;
}
.glyphicon-bitcoin:before {
  content: &quot;\e227&quot;;
}
.glyphicon-btc:before {
  content: &quot;\e227&quot;;
}
.glyphicon-xbt:before {
  content: &quot;\e227&quot;;
}
.glyphicon-yen:before {
  content: &quot;\00a5&quot;;
}
.glyphicon-jpy:before {
  content: &quot;\00a5&quot;;
}
.glyphicon-ruble:before {
  content: &quot;\20bd&quot;;
}
.glyphicon-rub:before {
  content: &quot;\20bd&quot;;
}
.glyphicon-scale:before {
  content: &quot;\e230&quot;;
}
.glyphicon-ice-lolly:before {
  content: &quot;\e231&quot;;
}
.glyphicon-ice-lolly-tasted:before {
  content: &quot;\e232&quot;;
}
.glyphicon-education:before {
  content: &quot;\e233&quot;;
}
.glyphicon-option-horizontal:before {
  content: &quot;\e234&quot;;
}
.glyphicon-option-vertical:before {
  content: &quot;\e235&quot;;
}
.glyphicon-menu-hamburger:before {
  content: &quot;\e236&quot;;
}
.glyphicon-modal-window:before {
  content: &quot;\e237&quot;;
}
.glyphicon-oil:before {
  content: &quot;\e238&quot;;
}
.glyphicon-grain:before {
  content: &quot;\e239&quot;;
}
.glyphicon-sunglasses:before {
  content: &quot;\e240&quot;;
}
.glyphicon-text-size:before {
  content: &quot;\e241&quot;;
}
.glyphicon-text-color:before {
  content: &quot;\e242&quot;;
}
.glyphicon-text-background:before {
  content: &quot;\e243&quot;;
}
.glyphicon-object-align-top:before {
  content: &quot;\e244&quot;;
}
.glyphicon-object-align-bottom:before {
  content: &quot;\e245&quot;;
}
.glyphicon-object-align-horizontal:before {
  content: &quot;\e246&quot;;
}
.glyphicon-object-align-left:before {
  content: &quot;\e247&quot;;
}
.glyphicon-object-align-vertical:before {
  content: &quot;\e248&quot;;
}
.glyphicon-object-align-right:before {
  content: &quot;\e249&quot;;
}
.glyphicon-triangle-right:before {
  content: &quot;\e250&quot;;
}
.glyphicon-triangle-left:before {
  content: &quot;\e251&quot;;
}
.glyphicon-triangle-bottom:before {
  content: &quot;\e252&quot;;
}
.glyphicon-triangle-top:before {
  content: &quot;\e253&quot;;
}
.glyphicon-console:before {
  content: &quot;\e254&quot;;
}
.glyphicon-superscript:before {
  content: &quot;\e255&quot;;
}
.glyphicon-subscript:before {
  content: &quot;\e256&quot;;
}
.glyphicon-menu-left:before {
  content: &quot;\e257&quot;;
}
.glyphicon-menu-right:before {
  content: &quot;\e258&quot;;
}
.glyphicon-menu-down:before {
  content: &quot;\e259&quot;;
}
.glyphicon-menu-up:before {
  content: &quot;\e260&quot;;
}
</em> {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
<em>:before,
</em>:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: &quot;Helvetica Neue&quot;, Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail &gt; img,
.thumbnail a &gt; img,
.carousel-inner &gt; .item &gt; img,
.carousel-inner &gt; .item &gt; a &gt; img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role=&quot;button&quot;] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline &gt; li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: &#39;\2014 \00A0&#39;;
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: &#39;&#39;;
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: &#39;\00A0 \2014&#39;;
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table &gt; thead &gt; tr &gt; th,
.table &gt; tbody &gt; tr &gt; th,
.table &gt; tfoot &gt; tr &gt; th,
.table &gt; thead &gt; tr &gt; td,
.table &gt; tbody &gt; tr &gt; td,
.table &gt; tfoot &gt; tr &gt; td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table &gt; thead &gt; tr &gt; th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table &gt; caption + thead &gt; tr:first-child &gt; th,
.table &gt; colgroup + thead &gt; tr:first-child &gt; th,
.table &gt; thead:first-child &gt; tr:first-child &gt; th,
.table &gt; caption + thead &gt; tr:first-child &gt; td,
.table &gt; colgroup + thead &gt; tr:first-child &gt; td,
.table &gt; thead:first-child &gt; tr:first-child &gt; td {
  border-top: 0;
}
.table &gt; tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed &gt; thead &gt; tr &gt; th,
.table-condensed &gt; tbody &gt; tr &gt; th,
.table-condensed &gt; tfoot &gt; tr &gt; th,
.table-condensed &gt; thead &gt; tr &gt; td,
.table-condensed &gt; tbody &gt; tr &gt; td,
.table-condensed &gt; tfoot &gt; tr &gt; td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered &gt; thead &gt; tr &gt; th,
.table-bordered &gt; tbody &gt; tr &gt; th,
.table-bordered &gt; tfoot &gt; tr &gt; th,
.table-bordered &gt; thead &gt; tr &gt; td,
.table-bordered &gt; tbody &gt; tr &gt; td,
.table-bordered &gt; tfoot &gt; tr &gt; td {
  border: 1px solid #ddd;
}
.table-bordered &gt; thead &gt; tr &gt; th,
.table-bordered &gt; thead &gt; tr &gt; td {
  border-bottom-width: 2px;
}
.table-striped &gt; tbody &gt; tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover &gt; tbody &gt; tr:hover {
  background-color: #f5f5f5;
}
table col[class<em>=&quot;col-&quot;] {
  position: static;
  float: none;
  display: table-column;
}
table td[class</em>=&quot;col-&quot;],
table th[class<em>=&quot;col-&quot;] {
  position: static;
  float: none;
  display: table-cell;
}
.table &gt; thead &gt; tr &gt; td.active,
.table &gt; tbody &gt; tr &gt; td.active,
.table &gt; tfoot &gt; tr &gt; td.active,
.table &gt; thead &gt; tr &gt; th.active,
.table &gt; tbody &gt; tr &gt; th.active,
.table &gt; tfoot &gt; tr &gt; th.active,
.table &gt; thead &gt; tr.active &gt; td,
.table &gt; tbody &gt; tr.active &gt; td,
.table &gt; tfoot &gt; tr.active &gt; td,
.table &gt; thead &gt; tr.active &gt; th,
.table &gt; tbody &gt; tr.active &gt; th,
.table &gt; tfoot &gt; tr.active &gt; th {
  background-color: #f5f5f5;
}
.table-hover &gt; tbody &gt; tr &gt; td.active:hover,
.table-hover &gt; tbody &gt; tr &gt; th.active:hover,
.table-hover &gt; tbody &gt; tr.active:hover &gt; td,
.table-hover &gt; tbody &gt; tr:hover &gt; .active,
.table-hover &gt; tbody &gt; tr.active:hover &gt; th {
  background-color: #e8e8e8;
}
.table &gt; thead &gt; tr &gt; td.success,
.table &gt; tbody &gt; tr &gt; td.success,
.table &gt; tfoot &gt; tr &gt; td.success,
.table &gt; thead &gt; tr &gt; th.success,
.table &gt; tbody &gt; tr &gt; th.success,
.table &gt; tfoot &gt; tr &gt; th.success,
.table &gt; thead &gt; tr.success &gt; td,
.table &gt; tbody &gt; tr.success &gt; td,
.table &gt; tfoot &gt; tr.success &gt; td,
.table &gt; thead &gt; tr.success &gt; th,
.table &gt; tbody &gt; tr.success &gt; th,
.table &gt; tfoot &gt; tr.success &gt; th {
  background-color: #dff0d8;
}
.table-hover &gt; tbody &gt; tr &gt; td.success:hover,
.table-hover &gt; tbody &gt; tr &gt; th.success:hover,
.table-hover &gt; tbody &gt; tr.success:hover &gt; td,
.table-hover &gt; tbody &gt; tr:hover &gt; .success,
.table-hover &gt; tbody &gt; tr.success:hover &gt; th {
  background-color: #d0e9c6;
}
.table &gt; thead &gt; tr &gt; td.info,
.table &gt; tbody &gt; tr &gt; td.info,
.table &gt; tfoot &gt; tr &gt; td.info,
.table &gt; thead &gt; tr &gt; th.info,
.table &gt; tbody &gt; tr &gt; th.info,
.table &gt; tfoot &gt; tr &gt; th.info,
.table &gt; thead &gt; tr.info &gt; td,
.table &gt; tbody &gt; tr.info &gt; td,
.table &gt; tfoot &gt; tr.info &gt; td,
.table &gt; thead &gt; tr.info &gt; th,
.table &gt; tbody &gt; tr.info &gt; th,
.table &gt; tfoot &gt; tr.info &gt; th {
  background-color: #d9edf7;
}
.table-hover &gt; tbody &gt; tr &gt; td.info:hover,
.table-hover &gt; tbody &gt; tr &gt; th.info:hover,
.table-hover &gt; tbody &gt; tr.info:hover &gt; td,
.table-hover &gt; tbody &gt; tr:hover &gt; .info,
.table-hover &gt; tbody &gt; tr.info:hover &gt; th {
  background-color: #c4e3f3;
}
.table &gt; thead &gt; tr &gt; td.warning,
.table &gt; tbody &gt; tr &gt; td.warning,
.table &gt; tfoot &gt; tr &gt; td.warning,
.table &gt; thead &gt; tr &gt; th.warning,
.table &gt; tbody &gt; tr &gt; th.warning,
.table &gt; tfoot &gt; tr &gt; th.warning,
.table &gt; thead &gt; tr.warning &gt; td,
.table &gt; tbody &gt; tr.warning &gt; td,
.table &gt; tfoot &gt; tr.warning &gt; td,
.table &gt; thead &gt; tr.warning &gt; th,
.table &gt; tbody &gt; tr.warning &gt; th,
.table &gt; tfoot &gt; tr.warning &gt; th {
  background-color: #fcf8e3;
}
.table-hover &gt; tbody &gt; tr &gt; td.warning:hover,
.table-hover &gt; tbody &gt; tr &gt; th.warning:hover,
.table-hover &gt; tbody &gt; tr.warning:hover &gt; td,
.table-hover &gt; tbody &gt; tr:hover &gt; .warning,
.table-hover &gt; tbody &gt; tr.warning:hover &gt; th {
  background-color: #faf2cc;
}
.table &gt; thead &gt; tr &gt; td.danger,
.table &gt; tbody &gt; tr &gt; td.danger,
.table &gt; tfoot &gt; tr &gt; td.danger,
.table &gt; thead &gt; tr &gt; th.danger,
.table &gt; tbody &gt; tr &gt; th.danger,
.table &gt; tfoot &gt; tr &gt; th.danger,
.table &gt; thead &gt; tr.danger &gt; td,
.table &gt; tbody &gt; tr.danger &gt; td,
.table &gt; tfoot &gt; tr.danger &gt; td,
.table &gt; thead &gt; tr.danger &gt; th,
.table &gt; tbody &gt; tr.danger &gt; th,
.table &gt; tfoot &gt; tr.danger &gt; th {
  background-color: #f2dede;
}
.table-hover &gt; tbody &gt; tr &gt; td.danger:hover,
.table-hover &gt; tbody &gt; tr &gt; th.danger:hover,
.table-hover &gt; tbody &gt; tr.danger:hover &gt; td,
.table-hover &gt; tbody &gt; tr:hover &gt; .danger,
.table-hover &gt; tbody &gt; tr.danger:hover &gt; th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive &gt; .table {
    margin-bottom: 0;
  }
  .table-responsive &gt; .table &gt; thead &gt; tr &gt; th,
  .table-responsive &gt; .table &gt; tbody &gt; tr &gt; th,
  .table-responsive &gt; .table &gt; tfoot &gt; tr &gt; th,
  .table-responsive &gt; .table &gt; thead &gt; tr &gt; td,
  .table-responsive &gt; .table &gt; tbody &gt; tr &gt; td,
  .table-responsive &gt; .table &gt; tfoot &gt; tr &gt; td {
    white-space: nowrap;
  }
  .table-responsive &gt; .table-bordered {
    border: 0;
  }
  .table-responsive &gt; .table-bordered &gt; thead &gt; tr &gt; th:first-child,
  .table-responsive &gt; .table-bordered &gt; tbody &gt; tr &gt; th:first-child,
  .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr &gt; th:first-child,
  .table-responsive &gt; .table-bordered &gt; thead &gt; tr &gt; td:first-child,
  .table-responsive &gt; .table-bordered &gt; tbody &gt; tr &gt; td:first-child,
  .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr &gt; td:first-child {
    border-left: 0;
  }
  .table-responsive &gt; .table-bordered &gt; thead &gt; tr &gt; th:last-child,
  .table-responsive &gt; .table-bordered &gt; tbody &gt; tr &gt; th:last-child,
  .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr &gt; th:last-child,
  .table-responsive &gt; .table-bordered &gt; thead &gt; tr &gt; td:last-child,
  .table-responsive &gt; .table-bordered &gt; tbody &gt; tr &gt; td:last-child,
  .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr &gt; td:last-child {
    border-right: 0;
  }
  .table-responsive &gt; .table-bordered &gt; tbody &gt; tr:last-child &gt; th,
  .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr:last-child &gt; th,
  .table-responsive &gt; .table-bordered &gt; tbody &gt; tr:last-child &gt; td,
  .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr:last-child &gt; td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type=&quot;search&quot;] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type=&quot;radio&quot;],
input[type=&quot;checkbox&quot;] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type=&quot;file&quot;] {
  display: block;
}
input[type=&quot;range&quot;] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type=&quot;file&quot;]:focus,
input[type=&quot;radio&quot;]:focus,
input[type=&quot;checkbox&quot;]:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type=&quot;search&quot;] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type=&quot;date&quot;].form-control,
  input[type=&quot;time&quot;].form-control,
  input[type=&quot;datetime-local&quot;].form-control,
  input[type=&quot;month&quot;].form-control {
    line-height: 32px;
  }
  input[type=&quot;date&quot;].input-sm,
  input[type=&quot;time&quot;].input-sm,
  input[type=&quot;datetime-local&quot;].input-sm,
  input[type=&quot;month&quot;].input-sm,
  .input-group-sm input[type=&quot;date&quot;],
  .input-group-sm input[type=&quot;time&quot;],
  .input-group-sm input[type=&quot;datetime-local&quot;],
  .input-group-sm input[type=&quot;month&quot;] {
    line-height: 30px;
  }
  input[type=&quot;date&quot;].input-lg,
  input[type=&quot;time&quot;].input-lg,
  input[type=&quot;datetime-local&quot;].input-lg,
  input[type=&quot;month&quot;].input-lg,
  .input-group-lg input[type=&quot;date&quot;],
  .input-group-lg input[type=&quot;time&quot;],
  .input-group-lg input[type=&quot;datetime-local&quot;],
  .input-group-lg input[type=&quot;month&quot;] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type=&quot;radio&quot;],
.radio-inline input[type=&quot;radio&quot;],
.checkbox input[type=&quot;checkbox&quot;],
.checkbox-inline input[type=&quot;checkbox&quot;] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type=&quot;radio&quot;][disabled],
input[type=&quot;checkbox&quot;][disabled],
input[type=&quot;radio&quot;].disabled,
input[type=&quot;checkbox&quot;].disabled,
fieldset[disabled] input[type=&quot;radio&quot;],
fieldset[disabled] input[type=&quot;checkbox&quot;] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group &gt; .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type=&quot;radio&quot;],
  .form-inline .checkbox input[type=&quot;checkbox&quot;] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open &gt; .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open &gt; .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open &gt; .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open &gt; .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open &gt; .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open &gt; .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open &gt; .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open &gt; .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open &gt; .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open &gt; .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open &gt; .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open &gt; .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open &gt; .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open &gt; .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open &gt; .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open &gt; .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open &gt; .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open &gt; .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open &gt; .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open &gt; .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open &gt; .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open &gt; .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open &gt; .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open &gt; .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open &gt; .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open &gt; .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open &gt; .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open &gt; .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open &gt; .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open &gt; .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg &gt; .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm &gt; .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs &gt; .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type=&quot;submit&quot;].btn-block,
input[type=&quot;reset&quot;].btn-block,
input[type=&quot;button&quot;].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu &gt; li &gt; a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu &gt; li &gt; a:hover,
.dropdown-menu &gt; li &gt; a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu &gt; .active &gt; a,
.dropdown-menu &gt; .active &gt; a:hover,
.dropdown-menu &gt; .active &gt; a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu &gt; .disabled &gt; a,
.dropdown-menu &gt; .disabled &gt; a:hover,
.dropdown-menu &gt; .disabled &gt; a:focus {
  color: #777777;
}
.dropdown-menu &gt; .disabled &gt; a:hover,
.dropdown-menu &gt; .disabled &gt; a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open &gt; .dropdown-menu {
  display: block;
}
.open &gt; a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right &gt; .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: &quot;&quot;;
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group &gt; .btn,
.btn-group-vertical &gt; .btn {
  position: relative;
  float: left;
}
.btn-group &gt; .btn:hover,
.btn-group-vertical &gt; .btn:hover,
.btn-group &gt; .btn:focus,
.btn-group-vertical &gt; .btn:focus,
.btn-group &gt; .btn:active,
.btn-group-vertical &gt; .btn:active,
.btn-group &gt; .btn.active,
.btn-group-vertical &gt; .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar &gt; .btn,
.btn-toolbar &gt; .btn-group,
.btn-toolbar &gt; .input-group {
  margin-left: 5px;
}
.btn-group &gt; .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group &gt; .btn:first-child {
  margin-left: 0;
}
.btn-group &gt; .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group &gt; .btn:last-child:not(:first-child),
.btn-group &gt; .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group &gt; .btn-group {
  float: left;
}
.btn-group &gt; .btn-group:not(:first-child):not(:last-child) &gt; .btn {
  border-radius: 0;
}
.btn-group &gt; .btn-group:first-child:not(:last-child) &gt; .btn:last-child,
.btn-group &gt; .btn-group:first-child:not(:last-child) &gt; .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group &gt; .btn-group:last-child:not(:first-child) &gt; .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group &gt; .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group &gt; .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical &gt; .btn,
.btn-group-vertical &gt; .btn-group,
.btn-group-vertical &gt; .btn-group &gt; .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical &gt; .btn-group &gt; .btn {
  float: none;
}
.btn-group-vertical &gt; .btn + .btn,
.btn-group-vertical &gt; .btn + .btn-group,
.btn-group-vertical &gt; .btn-group + .btn,
.btn-group-vertical &gt; .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical &gt; .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical &gt; .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical &gt; .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical &gt; .btn-group:not(:first-child):not(:last-child) &gt; .btn {
  border-radius: 0;
}
.btn-group-vertical &gt; .btn-group:first-child:not(:last-child) &gt; .btn:last-child,
.btn-group-vertical &gt; .btn-group:first-child:not(:last-child) &gt; .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical &gt; .btn-group:last-child:not(:first-child) &gt; .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified &gt; .btn,
.btn-group-justified &gt; .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified &gt; .btn-group .btn {
  width: 100%;
}
.btn-group-justified &gt; .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle=&quot;buttons&quot;] &gt; .btn input[type=&quot;radio&quot;],
[data-toggle=&quot;buttons&quot;] &gt; .btn-group &gt; .btn input[type=&quot;radio&quot;],
[data-toggle=&quot;buttons&quot;] &gt; .btn input[type=&quot;checkbox&quot;],
[data-toggle=&quot;buttons&quot;] &gt; .btn-group &gt; .btn input[type=&quot;checkbox&quot;] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class</em>=&quot;col-&quot;] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg &gt; .form-control,
.input-group-lg &gt; .input-group-addon,
.input-group-lg &gt; .input-group-btn &gt; .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg &gt; .form-control,
select.input-group-lg &gt; .input-group-addon,
select.input-group-lg &gt; .input-group-btn &gt; .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg &gt; .form-control,
textarea.input-group-lg &gt; .input-group-addon,
textarea.input-group-lg &gt; .input-group-btn &gt; .btn,
select[multiple].input-group-lg &gt; .form-control,
select[multiple].input-group-lg &gt; .input-group-addon,
select[multiple].input-group-lg &gt; .input-group-btn &gt; .btn {
  height: auto;
}
.input-group-sm &gt; .form-control,
.input-group-sm &gt; .input-group-addon,
.input-group-sm &gt; .input-group-btn &gt; .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm &gt; .form-control,
select.input-group-sm &gt; .input-group-addon,
select.input-group-sm &gt; .input-group-btn &gt; .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm &gt; .form-control,
textarea.input-group-sm &gt; .input-group-addon,
textarea.input-group-sm &gt; .input-group-btn &gt; .btn,
select[multiple].input-group-sm &gt; .form-control,
select[multiple].input-group-sm &gt; .input-group-addon,
select[multiple].input-group-sm &gt; .input-group-btn &gt; .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type=&quot;radio&quot;],
.input-group-addon input[type=&quot;checkbox&quot;] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child &gt; .btn,
.input-group-btn:first-child &gt; .btn-group &gt; .btn,
.input-group-btn:first-child &gt; .dropdown-toggle,
.input-group-btn:last-child &gt; .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child &gt; .btn-group:not(:last-child) &gt; .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child &gt; .btn,
.input-group-btn:last-child &gt; .btn-group &gt; .btn,
.input-group-btn:last-child &gt; .dropdown-toggle,
.input-group-btn:first-child &gt; .btn:not(:first-child),
.input-group-btn:first-child &gt; .btn-group:not(:first-child) &gt; .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn &gt; .btn {
  position: relative;
}
.input-group-btn &gt; .btn + .btn {
  margin-left: -1px;
}
.input-group-btn &gt; .btn:hover,
.input-group-btn &gt; .btn:focus,
.input-group-btn &gt; .btn:active {
  z-index: 2;
}
.input-group-btn:first-child &gt; .btn,
.input-group-btn:first-child &gt; .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child &gt; .btn,
.input-group-btn:last-child &gt; .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav &gt; li {
  position: relative;
  display: block;
}
.nav &gt; li &gt; a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav &gt; li &gt; a:hover,
.nav &gt; li &gt; a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav &gt; li.disabled &gt; a {
  color: #777777;
}
.nav &gt; li.disabled &gt; a:hover,
.nav &gt; li.disabled &gt; a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open &gt; a,
.nav .open &gt; a:hover,
.nav .open &gt; a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav &gt; li &gt; a &gt; img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs &gt; li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs &gt; li &gt; a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs &gt; li &gt; a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs &gt; li.active &gt; a,
.nav-tabs &gt; li.active &gt; a:hover,
.nav-tabs &gt; li.active &gt; a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified &gt; li {
  float: none;
}
.nav-tabs.nav-justified &gt; li &gt; a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified &gt; .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified &gt; li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified &gt; li &gt; a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified &gt; li &gt; a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified &gt; .active &gt; a,
.nav-tabs.nav-justified &gt; .active &gt; a:hover,
.nav-tabs.nav-justified &gt; .active &gt; a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified &gt; li &gt; a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified &gt; .active &gt; a,
  .nav-tabs.nav-justified &gt; .active &gt; a:hover,
  .nav-tabs.nav-justified &gt; .active &gt; a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills &gt; li {
  float: left;
}
.nav-pills &gt; li &gt; a {
  border-radius: 2px;
}
.nav-pills &gt; li + li {
  margin-left: 2px;
}
.nav-pills &gt; li.active &gt; a,
.nav-pills &gt; li.active &gt; a:hover,
.nav-pills &gt; li.active &gt; a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked &gt; li {
  float: none;
}
.nav-stacked &gt; li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified &gt; li {
  float: none;
}
.nav-justified &gt; li &gt; a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified &gt; .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified &gt; li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified &gt; li &gt; a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified &gt; li &gt; a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified &gt; .active &gt; a,
.nav-tabs-justified &gt; .active &gt; a:hover,
.nav-tabs-justified &gt; .active &gt; a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified &gt; li &gt; a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified &gt; .active &gt; a,
  .nav-tabs-justified &gt; .active &gt; a:hover,
  .nav-tabs-justified &gt; .active &gt; a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content &gt; .tab-pane {
  display: none;
}
.tab-content &gt; .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container &gt; .navbar-header,
.container-fluid &gt; .navbar-header,
.container &gt; .navbar-collapse,
.container-fluid &gt; .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container &gt; .navbar-header,
  .container-fluid &gt; .navbar-header,
  .container &gt; .navbar-collapse,
  .container-fluid &gt; .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand &gt; img {
  display: block;
}
@media (min-width: 541px) {
  .navbar &gt; .container .navbar-brand,
  .navbar &gt; .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav &gt; li &gt; a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu &gt; li &gt; a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu &gt; li &gt; a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu &gt; li &gt; a:hover,
  .navbar-nav .open .dropdown-menu &gt; li &gt; a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav &gt; li {
    float: left;
  }
  .navbar-nav &gt; li &gt; a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group &gt; .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type=&quot;radio&quot;],
  .navbar-form .checkbox input[type=&quot;checkbox&quot;] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav &gt; li &gt; .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav &gt; li &gt; .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav &gt; li &gt; a {
  color: #777;
}
.navbar-default .navbar-nav &gt; li &gt; a:hover,
.navbar-default .navbar-nav &gt; li &gt; a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav &gt; .active &gt; a,
.navbar-default .navbar-nav &gt; .active &gt; a:hover,
.navbar-default .navbar-nav &gt; .active &gt; a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav &gt; .disabled &gt; a,
.navbar-default .navbar-nav &gt; .disabled &gt; a:hover,
.navbar-default .navbar-nav &gt; .disabled &gt; a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav &gt; .open &gt; a,
.navbar-default .navbar-nav &gt; .open &gt; a:hover,
.navbar-default .navbar-nav &gt; .open &gt; a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu &gt; li &gt; a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu &gt; li &gt; a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu &gt; li &gt; a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu &gt; .active &gt; a,
  .navbar-default .navbar-nav .open .dropdown-menu &gt; .active &gt; a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu &gt; .active &gt; a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu &gt; .disabled &gt; a,
  .navbar-default .navbar-nav .open .dropdown-menu &gt; .disabled &gt; a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu &gt; .disabled &gt; a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav &gt; li &gt; a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav &gt; li &gt; a:hover,
.navbar-inverse .navbar-nav &gt; li &gt; a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav &gt; .active &gt; a,
.navbar-inverse .navbar-nav &gt; .active &gt; a:hover,
.navbar-inverse .navbar-nav &gt; .active &gt; a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav &gt; .disabled &gt; a,
.navbar-inverse .navbar-nav &gt; .disabled &gt; a:hover,
.navbar-inverse .navbar-nav &gt; .disabled &gt; a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav &gt; .open &gt; a,
.navbar-inverse .navbar-nav &gt; .open &gt; a:hover,
.navbar-inverse .navbar-nav &gt; .open &gt; a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu &gt; .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu &gt; li &gt; a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu &gt; li &gt; a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu &gt; li &gt; a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu &gt; .active &gt; a,
  .navbar-inverse .navbar-nav .open .dropdown-menu &gt; .active &gt; a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu &gt; .active &gt; a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu &gt; .disabled &gt; a,
  .navbar-inverse .navbar-nav .open .dropdown-menu &gt; .disabled &gt; a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu &gt; .disabled &gt; a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb &gt; li {
  display: inline-block;
}
.breadcrumb &gt; li + li:before {
  content: &quot;/\00a0&quot;;
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb &gt; .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination &gt; li {
  display: inline;
}
.pagination &gt; li &gt; a,
.pagination &gt; li &gt; span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination &gt; li:first-child &gt; a,
.pagination &gt; li:first-child &gt; span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination &gt; li:last-child &gt; a,
.pagination &gt; li:last-child &gt; span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination &gt; li &gt; a:hover,
.pagination &gt; li &gt; span:hover,
.pagination &gt; li &gt; a:focus,
.pagination &gt; li &gt; span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination &gt; .active &gt; a,
.pagination &gt; .active &gt; span,
.pagination &gt; .active &gt; a:hover,
.pagination &gt; .active &gt; span:hover,
.pagination &gt; .active &gt; a:focus,
.pagination &gt; .active &gt; span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination &gt; .disabled &gt; span,
.pagination &gt; .disabled &gt; span:hover,
.pagination &gt; .disabled &gt; span:focus,
.pagination &gt; .disabled &gt; a,
.pagination &gt; .disabled &gt; a:hover,
.pagination &gt; .disabled &gt; a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg &gt; li &gt; a,
.pagination-lg &gt; li &gt; span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg &gt; li:first-child &gt; a,
.pagination-lg &gt; li:first-child &gt; span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg &gt; li:last-child &gt; a,
.pagination-lg &gt; li:last-child &gt; span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm &gt; li &gt; a,
.pagination-sm &gt; li &gt; span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm &gt; li:first-child &gt; a,
.pagination-sm &gt; li:first-child &gt; span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm &gt; li:last-child &gt; a,
.pagination-sm &gt; li:last-child &gt; span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li &gt; a,
.pager li &gt; span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li &gt; a:hover,
.pager li &gt; a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next &gt; a,
.pager .next &gt; span {
  float: right;
}
.pager .previous &gt; a,
.pager .previous &gt; span {
  float: left;
}
.pager .disabled &gt; a,
.pager .disabled &gt; a:hover,
.pager .disabled &gt; a:focus,
.pager .disabled &gt; span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs &gt; .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active &gt; .badge,
.nav-pills &gt; .active &gt; a &gt; .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item &gt; .badge {
  float: right;
}
.list-group-item &gt; .badge + .badge {
  margin-right: 5px;
}
.nav-pills &gt; li &gt; a &gt; .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron &gt; hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail &gt; img,
.thumbnail a &gt; img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert &gt; p,
.alert &gt; ul {
  margin-bottom: 0;
}
.alert &gt; p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media &gt; .pull-right {
  padding-left: 10px;
}
.media-left,
.media &gt; .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading &gt; small,
.list-group-item.active:hover .list-group-item-heading &gt; small,
.list-group-item.active:focus .list-group-item-heading &gt; small,
.list-group-item.active .list-group-item-heading &gt; .small,
.list-group-item.active:hover .list-group-item-heading &gt; .small,
.list-group-item.active:focus .list-group-item-heading &gt; .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading &gt; .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title &gt; a,
.panel-title &gt; small,
.panel-title &gt; .small,
.panel-title &gt; small &gt; a,
.panel-title &gt; .small &gt; a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel &gt; .list-group,
.panel &gt; .panel-collapse &gt; .list-group {
  margin-bottom: 0;
}
.panel &gt; .list-group .list-group-item,
.panel &gt; .panel-collapse &gt; .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel &gt; .list-group:first-child .list-group-item:first-child,
.panel &gt; .panel-collapse &gt; .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel &gt; .list-group:last-child .list-group-item:last-child,
.panel &gt; .panel-collapse &gt; .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel &gt; .panel-heading + .panel-collapse &gt; .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel &gt; .table,
.panel &gt; .table-responsive &gt; .table,
.panel &gt; .panel-collapse &gt; .table {
  margin-bottom: 0;
}
.panel &gt; .table caption,
.panel &gt; .table-responsive &gt; .table caption,
.panel &gt; .panel-collapse &gt; .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel &gt; .table:first-child,
.panel &gt; .table-responsive:first-child &gt; .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel &gt; .table:first-child &gt; thead:first-child &gt; tr:first-child,
.panel &gt; .table-responsive:first-child &gt; .table:first-child &gt; thead:first-child &gt; tr:first-child,
.panel &gt; .table:first-child &gt; tbody:first-child &gt; tr:first-child,
.panel &gt; .table-responsive:first-child &gt; .table:first-child &gt; tbody:first-child &gt; tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel &gt; .table:first-child &gt; thead:first-child &gt; tr:first-child td:first-child,
.panel &gt; .table-responsive:first-child &gt; .table:first-child &gt; thead:first-child &gt; tr:first-child td:first-child,
.panel &gt; .table:first-child &gt; tbody:first-child &gt; tr:first-child td:first-child,
.panel &gt; .table-responsive:first-child &gt; .table:first-child &gt; tbody:first-child &gt; tr:first-child td:first-child,
.panel &gt; .table:first-child &gt; thead:first-child &gt; tr:first-child th:first-child,
.panel &gt; .table-responsive:first-child &gt; .table:first-child &gt; thead:first-child &gt; tr:first-child th:first-child,
.panel &gt; .table:first-child &gt; tbody:first-child &gt; tr:first-child th:first-child,
.panel &gt; .table-responsive:first-child &gt; .table:first-child &gt; tbody:first-child &gt; tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel &gt; .table:first-child &gt; thead:first-child &gt; tr:first-child td:last-child,
.panel &gt; .table-responsive:first-child &gt; .table:first-child &gt; thead:first-child &gt; tr:first-child td:last-child,
.panel &gt; .table:first-child &gt; tbody:first-child &gt; tr:first-child td:last-child,
.panel &gt; .table-responsive:first-child &gt; .table:first-child &gt; tbody:first-child &gt; tr:first-child td:last-child,
.panel &gt; .table:first-child &gt; thead:first-child &gt; tr:first-child th:last-child,
.panel &gt; .table-responsive:first-child &gt; .table:first-child &gt; thead:first-child &gt; tr:first-child th:last-child,
.panel &gt; .table:first-child &gt; tbody:first-child &gt; tr:first-child th:last-child,
.panel &gt; .table-responsive:first-child &gt; .table:first-child &gt; tbody:first-child &gt; tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel &gt; .table:last-child,
.panel &gt; .table-responsive:last-child &gt; .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel &gt; .table:last-child &gt; tbody:last-child &gt; tr:last-child,
.panel &gt; .table-responsive:last-child &gt; .table:last-child &gt; tbody:last-child &gt; tr:last-child,
.panel &gt; .table:last-child &gt; tfoot:last-child &gt; tr:last-child,
.panel &gt; .table-responsive:last-child &gt; .table:last-child &gt; tfoot:last-child &gt; tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel &gt; .table:last-child &gt; tbody:last-child &gt; tr:last-child td:first-child,
.panel &gt; .table-responsive:last-child &gt; .table:last-child &gt; tbody:last-child &gt; tr:last-child td:first-child,
.panel &gt; .table:last-child &gt; tfoot:last-child &gt; tr:last-child td:first-child,
.panel &gt; .table-responsive:last-child &gt; .table:last-child &gt; tfoot:last-child &gt; tr:last-child td:first-child,
.panel &gt; .table:last-child &gt; tbody:last-child &gt; tr:last-child th:first-child,
.panel &gt; .table-responsive:last-child &gt; .table:last-child &gt; tbody:last-child &gt; tr:last-child th:first-child,
.panel &gt; .table:last-child &gt; tfoot:last-child &gt; tr:last-child th:first-child,
.panel &gt; .table-responsive:last-child &gt; .table:last-child &gt; tfoot:last-child &gt; tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel &gt; .table:last-child &gt; tbody:last-child &gt; tr:last-child td:last-child,
.panel &gt; .table-responsive:last-child &gt; .table:last-child &gt; tbody:last-child &gt; tr:last-child td:last-child,
.panel &gt; .table:last-child &gt; tfoot:last-child &gt; tr:last-child td:last-child,
.panel &gt; .table-responsive:last-child &gt; .table:last-child &gt; tfoot:last-child &gt; tr:last-child td:last-child,
.panel &gt; .table:last-child &gt; tbody:last-child &gt; tr:last-child th:last-child,
.panel &gt; .table-responsive:last-child &gt; .table:last-child &gt; tbody:last-child &gt; tr:last-child th:last-child,
.panel &gt; .table:last-child &gt; tfoot:last-child &gt; tr:last-child th:last-child,
.panel &gt; .table-responsive:last-child &gt; .table:last-child &gt; tfoot:last-child &gt; tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel &gt; .panel-body + .table,
.panel &gt; .panel-body + .table-responsive,
.panel &gt; .table + .panel-body,
.panel &gt; .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel &gt; .table &gt; tbody:first-child &gt; tr:first-child th,
.panel &gt; .table &gt; tbody:first-child &gt; tr:first-child td {
  border-top: 0;
}
.panel &gt; .table-bordered,
.panel &gt; .table-responsive &gt; .table-bordered {
  border: 0;
}
.panel &gt; .table-bordered &gt; thead &gt; tr &gt; th:first-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; thead &gt; tr &gt; th:first-child,
.panel &gt; .table-bordered &gt; tbody &gt; tr &gt; th:first-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tbody &gt; tr &gt; th:first-child,
.panel &gt; .table-bordered &gt; tfoot &gt; tr &gt; th:first-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr &gt; th:first-child,
.panel &gt; .table-bordered &gt; thead &gt; tr &gt; td:first-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; thead &gt; tr &gt; td:first-child,
.panel &gt; .table-bordered &gt; tbody &gt; tr &gt; td:first-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tbody &gt; tr &gt; td:first-child,
.panel &gt; .table-bordered &gt; tfoot &gt; tr &gt; td:first-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr &gt; td:first-child {
  border-left: 0;
}
.panel &gt; .table-bordered &gt; thead &gt; tr &gt; th:last-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; thead &gt; tr &gt; th:last-child,
.panel &gt; .table-bordered &gt; tbody &gt; tr &gt; th:last-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tbody &gt; tr &gt; th:last-child,
.panel &gt; .table-bordered &gt; tfoot &gt; tr &gt; th:last-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr &gt; th:last-child,
.panel &gt; .table-bordered &gt; thead &gt; tr &gt; td:last-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; thead &gt; tr &gt; td:last-child,
.panel &gt; .table-bordered &gt; tbody &gt; tr &gt; td:last-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tbody &gt; tr &gt; td:last-child,
.panel &gt; .table-bordered &gt; tfoot &gt; tr &gt; td:last-child,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr &gt; td:last-child {
  border-right: 0;
}
.panel &gt; .table-bordered &gt; thead &gt; tr:first-child &gt; td,
.panel &gt; .table-responsive &gt; .table-bordered &gt; thead &gt; tr:first-child &gt; td,
.panel &gt; .table-bordered &gt; tbody &gt; tr:first-child &gt; td,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tbody &gt; tr:first-child &gt; td,
.panel &gt; .table-bordered &gt; thead &gt; tr:first-child &gt; th,
.panel &gt; .table-responsive &gt; .table-bordered &gt; thead &gt; tr:first-child &gt; th,
.panel &gt; .table-bordered &gt; tbody &gt; tr:first-child &gt; th,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tbody &gt; tr:first-child &gt; th {
  border-bottom: 0;
}
.panel &gt; .table-bordered &gt; tbody &gt; tr:last-child &gt; td,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tbody &gt; tr:last-child &gt; td,
.panel &gt; .table-bordered &gt; tfoot &gt; tr:last-child &gt; td,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr:last-child &gt; td,
.panel &gt; .table-bordered &gt; tbody &gt; tr:last-child &gt; th,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tbody &gt; tr:last-child &gt; th,
.panel &gt; .table-bordered &gt; tfoot &gt; tr:last-child &gt; th,
.panel &gt; .table-responsive &gt; .table-bordered &gt; tfoot &gt; tr:last-child &gt; th {
  border-bottom: 0;
}
.panel &gt; .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse &gt; .panel-body,
.panel-group .panel-heading + .panel-collapse &gt; .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default &gt; .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default &gt; .panel-heading + .panel-collapse &gt; .panel-body {
  border-top-color: #ddd;
}
.panel-default &gt; .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default &gt; .panel-footer + .panel-collapse &gt; .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary &gt; .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary &gt; .panel-heading + .panel-collapse &gt; .panel-body {
  border-top-color: #337ab7;
}
.panel-primary &gt; .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary &gt; .panel-footer + .panel-collapse &gt; .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success &gt; .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success &gt; .panel-heading + .panel-collapse &gt; .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success &gt; .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success &gt; .panel-footer + .panel-collapse &gt; .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info &gt; .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info &gt; .panel-heading + .panel-collapse &gt; .panel-body {
  border-top-color: #bce8f1;
}
.panel-info &gt; .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info &gt; .panel-footer + .panel-collapse &gt; .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning &gt; .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning &gt; .panel-heading + .panel-collapse &gt; .panel-body {
  border-top-color: #faebcc;
}
.panel-warning &gt; .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning &gt; .panel-footer + .panel-collapse &gt; .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger &gt; .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger &gt; .panel-heading + .panel-collapse &gt; .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger &gt; .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger &gt; .panel-footer + .panel-collapse &gt; .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: &quot;Helvetica Neue&quot;, Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: &quot;Helvetica Neue&quot;, Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover &gt; .arrow,
.popover &gt; .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover &gt; .arrow {
  border-width: 11px;
}
.popover &gt; .arrow:after {
  border-width: 10px;
  content: &quot;&quot;;
}
.popover.top &gt; .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top &gt; .arrow:after {
  content: &quot; &quot;;
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right &gt; .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right &gt; .arrow:after {
  content: &quot; &quot;;
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom &gt; .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom &gt; .arrow:after {
  content: &quot; &quot;;
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left &gt; .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left &gt; .arrow:after {
  content: &quot; &quot;;
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner &gt; .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner &gt; .item &gt; img,
.carousel-inner &gt; .item &gt; a &gt; img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner &gt; .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner &gt; .item.next,
  .carousel-inner &gt; .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner &gt; .item.prev,
  .carousel-inner &gt; .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner &gt; .item.next.left,
  .carousel-inner &gt; .item.prev.right,
  .carousel-inner &gt; .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner &gt; .active,
.carousel-inner &gt; .next,
.carousel-inner &gt; .prev {
  display: block;
}
.carousel-inner &gt; .active {
  left: 0;
}
.carousel-inner &gt; .next,
.carousel-inner &gt; .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner &gt; .next {
  left: 100%;
}
.carousel-inner &gt; .prev {
  left: -100%;
}
.carousel-inner &gt; .next.left,
.carousel-inner &gt; .prev.right {
  left: 0;
}
.carousel-inner &gt; .active.left {
  left: -100%;
}
.carousel-inner &gt; .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr=&#39;#80000000&#39;, endColorstr=&#39;#00000000&#39;, GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr=&#39;#00000000&#39;, endColorstr=&#39;#80000000&#39;, GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: &#39;\2039&#39;;
}
.carousel-control .icon-next:before {
  content: &#39;\203a&#39;;
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical &gt; .btn-group:before,
.btn-group-vertical &gt; .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: &quot; &quot;;
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical &gt; .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/<em>!
</em>
<em> Font Awesome
</em>
<em>/
/</em>!
 <em>  Font Awesome 4.2.0 by @davegandy - <a href="http://fontawesome.io">http://fontawesome.io</a> - @fontawesome
 </em>  License - <a href="http://fontawesome.io/license">http://fontawesome.io/license</a> (Font: SIL OFL 1.1, CSS: MIT License)
 <em>/
/</em> FONT PATH
 <em> -------------------------- </em>/
@font-face {
  font-family: &#39;FontAwesome&#39;;
  src: url(&#39;../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.2.0&#39;);
  src: url(&#39;../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&amp;v=4.2.0&#39;) format(&#39;embedded-opentype&#39;), url(&#39;../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.2.0&#39;) format(&#39;woff&#39;), url(&#39;../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.2.0&#39;) format(&#39;truetype&#39;), url(&#39;../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.2.0#fontawesomeregular&#39;) format(&#39;svg&#39;);
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/<em> makes the font 33% larger relative to the icon container </em>/
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul &gt; li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=1);
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2);
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=3);
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1);
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1);
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/<em> Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons </em>/
.fa-glass:before {
  content: &quot;\f000&quot;;
}
.fa-music:before {
  content: &quot;\f001&quot;;
}
.fa-search:before {
  content: &quot;\f002&quot;;
}
.fa-envelope-o:before {
  content: &quot;\f003&quot;;
}
.fa-heart:before {
  content: &quot;\f004&quot;;
}
.fa-star:before {
  content: &quot;\f005&quot;;
}
.fa-star-o:before {
  content: &quot;\f006&quot;;
}
.fa-user:before {
  content: &quot;\f007&quot;;
}
.fa-film:before {
  content: &quot;\f008&quot;;
}
.fa-th-large:before {
  content: &quot;\f009&quot;;
}
.fa-th:before {
  content: &quot;\f00a&quot;;
}
.fa-th-list:before {
  content: &quot;\f00b&quot;;
}
.fa-check:before {
  content: &quot;\f00c&quot;;
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: &quot;\f00d&quot;;
}
.fa-search-plus:before {
  content: &quot;\f00e&quot;;
}
.fa-search-minus:before {
  content: &quot;\f010&quot;;
}
.fa-power-off:before {
  content: &quot;\f011&quot;;
}
.fa-signal:before {
  content: &quot;\f012&quot;;
}
.fa-gear:before,
.fa-cog:before {
  content: &quot;\f013&quot;;
}
.fa-trash-o:before {
  content: &quot;\f014&quot;;
}
.fa-home:before {
  content: &quot;\f015&quot;;
}
.fa-file-o:before {
  content: &quot;\f016&quot;;
}
.fa-clock-o:before {
  content: &quot;\f017&quot;;
}
.fa-road:before {
  content: &quot;\f018&quot;;
}
.fa-download:before {
  content: &quot;\f019&quot;;
}
.fa-arrow-circle-o-down:before {
  content: &quot;\f01a&quot;;
}
.fa-arrow-circle-o-up:before {
  content: &quot;\f01b&quot;;
}
.fa-inbox:before {
  content: &quot;\f01c&quot;;
}
.fa-play-circle-o:before {
  content: &quot;\f01d&quot;;
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: &quot;\f01e&quot;;
}
.fa-refresh:before {
  content: &quot;\f021&quot;;
}
.fa-list-alt:before {
  content: &quot;\f022&quot;;
}
.fa-lock:before {
  content: &quot;\f023&quot;;
}
.fa-flag:before {
  content: &quot;\f024&quot;;
}
.fa-headphones:before {
  content: &quot;\f025&quot;;
}
.fa-volume-off:before {
  content: &quot;\f026&quot;;
}
.fa-volume-down:before {
  content: &quot;\f027&quot;;
}
.fa-volume-up:before {
  content: &quot;\f028&quot;;
}
.fa-qrcode:before {
  content: &quot;\f029&quot;;
}
.fa-barcode:before {
  content: &quot;\f02a&quot;;
}
.fa-tag:before {
  content: &quot;\f02b&quot;;
}
.fa-tags:before {
  content: &quot;\f02c&quot;;
}
.fa-book:before {
  content: &quot;\f02d&quot;;
}
.fa-bookmark:before {
  content: &quot;\f02e&quot;;
}
.fa-print:before {
  content: &quot;\f02f&quot;;
}
.fa-camera:before {
  content: &quot;\f030&quot;;
}
.fa-font:before {
  content: &quot;\f031&quot;;
}
.fa-bold:before {
  content: &quot;\f032&quot;;
}
.fa-italic:before {
  content: &quot;\f033&quot;;
}
.fa-text-height:before {
  content: &quot;\f034&quot;;
}
.fa-text-width:before {
  content: &quot;\f035&quot;;
}
.fa-align-left:before {
  content: &quot;\f036&quot;;
}
.fa-align-center:before {
  content: &quot;\f037&quot;;
}
.fa-align-right:before {
  content: &quot;\f038&quot;;
}
.fa-align-justify:before {
  content: &quot;\f039&quot;;
}
.fa-list:before {
  content: &quot;\f03a&quot;;
}
.fa-dedent:before,
.fa-outdent:before {
  content: &quot;\f03b&quot;;
}
.fa-indent:before {
  content: &quot;\f03c&quot;;
}
.fa-video-camera:before {
  content: &quot;\f03d&quot;;
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: &quot;\f03e&quot;;
}
.fa-pencil:before {
  content: &quot;\f040&quot;;
}
.fa-map-marker:before {
  content: &quot;\f041&quot;;
}
.fa-adjust:before {
  content: &quot;\f042&quot;;
}
.fa-tint:before {
  content: &quot;\f043&quot;;
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: &quot;\f044&quot;;
}
.fa-share-square-o:before {
  content: &quot;\f045&quot;;
}
.fa-check-square-o:before {
  content: &quot;\f046&quot;;
}
.fa-arrows:before {
  content: &quot;\f047&quot;;
}
.fa-step-backward:before {
  content: &quot;\f048&quot;;
}
.fa-fast-backward:before {
  content: &quot;\f049&quot;;
}
.fa-backward:before {
  content: &quot;\f04a&quot;;
}
.fa-play:before {
  content: &quot;\f04b&quot;;
}
.fa-pause:before {
  content: &quot;\f04c&quot;;
}
.fa-stop:before {
  content: &quot;\f04d&quot;;
}
.fa-forward:before {
  content: &quot;\f04e&quot;;
}
.fa-fast-forward:before {
  content: &quot;\f050&quot;;
}
.fa-step-forward:before {
  content: &quot;\f051&quot;;
}
.fa-eject:before {
  content: &quot;\f052&quot;;
}
.fa-chevron-left:before {
  content: &quot;\f053&quot;;
}
.fa-chevron-right:before {
  content: &quot;\f054&quot;;
}
.fa-plus-circle:before {
  content: &quot;\f055&quot;;
}
.fa-minus-circle:before {
  content: &quot;\f056&quot;;
}
.fa-times-circle:before {
  content: &quot;\f057&quot;;
}
.fa-check-circle:before {
  content: &quot;\f058&quot;;
}
.fa-question-circle:before {
  content: &quot;\f059&quot;;
}
.fa-info-circle:before {
  content: &quot;\f05a&quot;;
}
.fa-crosshairs:before {
  content: &quot;\f05b&quot;;
}
.fa-times-circle-o:before {
  content: &quot;\f05c&quot;;
}
.fa-check-circle-o:before {
  content: &quot;\f05d&quot;;
}
.fa-ban:before {
  content: &quot;\f05e&quot;;
}
.fa-arrow-left:before {
  content: &quot;\f060&quot;;
}
.fa-arrow-right:before {
  content: &quot;\f061&quot;;
}
.fa-arrow-up:before {
  content: &quot;\f062&quot;;
}
.fa-arrow-down:before {
  content: &quot;\f063&quot;;
}
.fa-mail-forward:before,
.fa-share:before {
  content: &quot;\f064&quot;;
}
.fa-expand:before {
  content: &quot;\f065&quot;;
}
.fa-compress:before {
  content: &quot;\f066&quot;;
}
.fa-plus:before {
  content: &quot;\f067&quot;;
}
.fa-minus:before {
  content: &quot;\f068&quot;;
}
.fa-asterisk:before {
  content: &quot;\f069&quot;;
}
.fa-exclamation-circle:before {
  content: &quot;\f06a&quot;;
}
.fa-gift:before {
  content: &quot;\f06b&quot;;
}
.fa-leaf:before {
  content: &quot;\f06c&quot;;
}
.fa-fire:before {
  content: &quot;\f06d&quot;;
}
.fa-eye:before {
  content: &quot;\f06e&quot;;
}
.fa-eye-slash:before {
  content: &quot;\f070&quot;;
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: &quot;\f071&quot;;
}
.fa-plane:before {
  content: &quot;\f072&quot;;
}
.fa-calendar:before {
  content: &quot;\f073&quot;;
}
.fa-random:before {
  content: &quot;\f074&quot;;
}
.fa-comment:before {
  content: &quot;\f075&quot;;
}
.fa-magnet:before {
  content: &quot;\f076&quot;;
}
.fa-chevron-up:before {
  content: &quot;\f077&quot;;
}
.fa-chevron-down:before {
  content: &quot;\f078&quot;;
}
.fa-retweet:before {
  content: &quot;\f079&quot;;
}
.fa-shopping-cart:before {
  content: &quot;\f07a&quot;;
}
.fa-folder:before {
  content: &quot;\f07b&quot;;
}
.fa-folder-open:before {
  content: &quot;\f07c&quot;;
}
.fa-arrows-v:before {
  content: &quot;\f07d&quot;;
}
.fa-arrows-h:before {
  content: &quot;\f07e&quot;;
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: &quot;\f080&quot;;
}
.fa-twitter-square:before {
  content: &quot;\f081&quot;;
}
.fa-facebook-square:before {
  content: &quot;\f082&quot;;
}
.fa-camera-retro:before {
  content: &quot;\f083&quot;;
}
.fa-key:before {
  content: &quot;\f084&quot;;
}
.fa-gears:before,
.fa-cogs:before {
  content: &quot;\f085&quot;;
}
.fa-comments:before {
  content: &quot;\f086&quot;;
}
.fa-thumbs-o-up:before {
  content: &quot;\f087&quot;;
}
.fa-thumbs-o-down:before {
  content: &quot;\f088&quot;;
}
.fa-star-half:before {
  content: &quot;\f089&quot;;
}
.fa-heart-o:before {
  content: &quot;\f08a&quot;;
}
.fa-sign-out:before {
  content: &quot;\f08b&quot;;
}
.fa-linkedin-square:before {
  content: &quot;\f08c&quot;;
}
.fa-thumb-tack:before {
  content: &quot;\f08d&quot;;
}
.fa-external-link:before {
  content: &quot;\f08e&quot;;
}
.fa-sign-in:before {
  content: &quot;\f090&quot;;
}
.fa-trophy:before {
  content: &quot;\f091&quot;;
}
.fa-github-square:before {
  content: &quot;\f092&quot;;
}
.fa-upload:before {
  content: &quot;\f093&quot;;
}
.fa-lemon-o:before {
  content: &quot;\f094&quot;;
}
.fa-phone:before {
  content: &quot;\f095&quot;;
}
.fa-square-o:before {
  content: &quot;\f096&quot;;
}
.fa-bookmark-o:before {
  content: &quot;\f097&quot;;
}
.fa-phone-square:before {
  content: &quot;\f098&quot;;
}
.fa-twitter:before {
  content: &quot;\f099&quot;;
}
.fa-facebook:before {
  content: &quot;\f09a&quot;;
}
.fa-github:before {
  content: &quot;\f09b&quot;;
}
.fa-unlock:before {
  content: &quot;\f09c&quot;;
}
.fa-credit-card:before {
  content: &quot;\f09d&quot;;
}
.fa-rss:before {
  content: &quot;\f09e&quot;;
}
.fa-hdd-o:before {
  content: &quot;\f0a0&quot;;
}
.fa-bullhorn:before {
  content: &quot;\f0a1&quot;;
}
.fa-bell:before {
  content: &quot;\f0f3&quot;;
}
.fa-certificate:before {
  content: &quot;\f0a3&quot;;
}
.fa-hand-o-right:before {
  content: &quot;\f0a4&quot;;
}
.fa-hand-o-left:before {
  content: &quot;\f0a5&quot;;
}
.fa-hand-o-up:before {
  content: &quot;\f0a6&quot;;
}
.fa-hand-o-down:before {
  content: &quot;\f0a7&quot;;
}
.fa-arrow-circle-left:before {
  content: &quot;\f0a8&quot;;
}
.fa-arrow-circle-right:before {
  content: &quot;\f0a9&quot;;
}
.fa-arrow-circle-up:before {
  content: &quot;\f0aa&quot;;
}
.fa-arrow-circle-down:before {
  content: &quot;\f0ab&quot;;
}
.fa-globe:before {
  content: &quot;\f0ac&quot;;
}
.fa-wrench:before {
  content: &quot;\f0ad&quot;;
}
.fa-tasks:before {
  content: &quot;\f0ae&quot;;
}
.fa-filter:before {
  content: &quot;\f0b0&quot;;
}
.fa-briefcase:before {
  content: &quot;\f0b1&quot;;
}
.fa-arrows-alt:before {
  content: &quot;\f0b2&quot;;
}
.fa-group:before,
.fa-users:before {
  content: &quot;\f0c0&quot;;
}
.fa-chain:before,
.fa-link:before {
  content: &quot;\f0c1&quot;;
}
.fa-cloud:before {
  content: &quot;\f0c2&quot;;
}
.fa-flask:before {
  content: &quot;\f0c3&quot;;
}
.fa-cut:before,
.fa-scissors:before {
  content: &quot;\f0c4&quot;;
}
.fa-copy:before,
.fa-files-o:before {
  content: &quot;\f0c5&quot;;
}
.fa-paperclip:before {
  content: &quot;\f0c6&quot;;
}
.fa-save:before,
.fa-floppy-o:before {
  content: &quot;\f0c7&quot;;
}
.fa-square:before {
  content: &quot;\f0c8&quot;;
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: &quot;\f0c9&quot;;
}
.fa-list-ul:before {
  content: &quot;\f0ca&quot;;
}
.fa-list-ol:before {
  content: &quot;\f0cb&quot;;
}
.fa-strikethrough:before {
  content: &quot;\f0cc&quot;;
}
.fa-underline:before {
  content: &quot;\f0cd&quot;;
}
.fa-table:before {
  content: &quot;\f0ce&quot;;
}
.fa-magic:before {
  content: &quot;\f0d0&quot;;
}
.fa-truck:before {
  content: &quot;\f0d1&quot;;
}
.fa-pinterest:before {
  content: &quot;\f0d2&quot;;
}
.fa-pinterest-square:before {
  content: &quot;\f0d3&quot;;
}
.fa-google-plus-square:before {
  content: &quot;\f0d4&quot;;
}
.fa-google-plus:before {
  content: &quot;\f0d5&quot;;
}
.fa-money:before {
  content: &quot;\f0d6&quot;;
}
.fa-caret-down:before {
  content: &quot;\f0d7&quot;;
}
.fa-caret-up:before {
  content: &quot;\f0d8&quot;;
}
.fa-caret-left:before {
  content: &quot;\f0d9&quot;;
}
.fa-caret-right:before {
  content: &quot;\f0da&quot;;
}
.fa-columns:before {
  content: &quot;\f0db&quot;;
}
.fa-unsorted:before,
.fa-sort:before {
  content: &quot;\f0dc&quot;;
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: &quot;\f0dd&quot;;
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: &quot;\f0de&quot;;
}
.fa-envelope:before {
  content: &quot;\f0e0&quot;;
}
.fa-linkedin:before {
  content: &quot;\f0e1&quot;;
}
.fa-rotate-left:before,
.fa-undo:before {
  content: &quot;\f0e2&quot;;
}
.fa-legal:before,
.fa-gavel:before {
  content: &quot;\f0e3&quot;;
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: &quot;\f0e4&quot;;
}
.fa-comment-o:before {
  content: &quot;\f0e5&quot;;
}
.fa-comments-o:before {
  content: &quot;\f0e6&quot;;
}
.fa-flash:before,
.fa-bolt:before {
  content: &quot;\f0e7&quot;;
}
.fa-sitemap:before {
  content: &quot;\f0e8&quot;;
}
.fa-umbrella:before {
  content: &quot;\f0e9&quot;;
}
.fa-paste:before,
.fa-clipboard:before {
  content: &quot;\f0ea&quot;;
}
.fa-lightbulb-o:before {
  content: &quot;\f0eb&quot;;
}
.fa-exchange:before {
  content: &quot;\f0ec&quot;;
}
.fa-cloud-download:before {
  content: &quot;\f0ed&quot;;
}
.fa-cloud-upload:before {
  content: &quot;\f0ee&quot;;
}
.fa-user-md:before {
  content: &quot;\f0f0&quot;;
}
.fa-stethoscope:before {
  content: &quot;\f0f1&quot;;
}
.fa-suitcase:before {
  content: &quot;\f0f2&quot;;
}
.fa-bell-o:before {
  content: &quot;\f0a2&quot;;
}
.fa-coffee:before {
  content: &quot;\f0f4&quot;;
}
.fa-cutlery:before {
  content: &quot;\f0f5&quot;;
}
.fa-file-text-o:before {
  content: &quot;\f0f6&quot;;
}
.fa-building-o:before {
  content: &quot;\f0f7&quot;;
}
.fa-hospital-o:before {
  content: &quot;\f0f8&quot;;
}
.fa-ambulance:before {
  content: &quot;\f0f9&quot;;
}
.fa-medkit:before {
  content: &quot;\f0fa&quot;;
}
.fa-fighter-jet:before {
  content: &quot;\f0fb&quot;;
}
.fa-beer:before {
  content: &quot;\f0fc&quot;;
}
.fa-h-square:before {
  content: &quot;\f0fd&quot;;
}
.fa-plus-square:before {
  content: &quot;\f0fe&quot;;
}
.fa-angle-double-left:before {
  content: &quot;\f100&quot;;
}
.fa-angle-double-right:before {
  content: &quot;\f101&quot;;
}
.fa-angle-double-up:before {
  content: &quot;\f102&quot;;
}
.fa-angle-double-down:before {
  content: &quot;\f103&quot;;
}
.fa-angle-left:before {
  content: &quot;\f104&quot;;
}
.fa-angle-right:before {
  content: &quot;\f105&quot;;
}
.fa-angle-up:before {
  content: &quot;\f106&quot;;
}
.fa-angle-down:before {
  content: &quot;\f107&quot;;
}
.fa-desktop:before {
  content: &quot;\f108&quot;;
}
.fa-laptop:before {
  content: &quot;\f109&quot;;
}
.fa-tablet:before {
  content: &quot;\f10a&quot;;
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: &quot;\f10b&quot;;
}
.fa-circle-o:before {
  content: &quot;\f10c&quot;;
}
.fa-quote-left:before {
  content: &quot;\f10d&quot;;
}
.fa-quote-right:before {
  content: &quot;\f10e&quot;;
}
.fa-spinner:before {
  content: &quot;\f110&quot;;
}
.fa-circle:before {
  content: &quot;\f111&quot;;
}
.fa-mail-reply:before,
.fa-reply:before {
  content: &quot;\f112&quot;;
}
.fa-github-alt:before {
  content: &quot;\f113&quot;;
}
.fa-folder-o:before {
  content: &quot;\f114&quot;;
}
.fa-folder-open-o:before {
  content: &quot;\f115&quot;;
}
.fa-smile-o:before {
  content: &quot;\f118&quot;;
}
.fa-frown-o:before {
  content: &quot;\f119&quot;;
}
.fa-meh-o:before {
  content: &quot;\f11a&quot;;
}
.fa-gamepad:before {
  content: &quot;\f11b&quot;;
}
.fa-keyboard-o:before {
  content: &quot;\f11c&quot;;
}
.fa-flag-o:before {
  content: &quot;\f11d&quot;;
}
.fa-flag-checkered:before {
  content: &quot;\f11e&quot;;
}
.fa-terminal:before {
  content: &quot;\f120&quot;;
}
.fa-code:before {
  content: &quot;\f121&quot;;
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: &quot;\f122&quot;;
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: &quot;\f123&quot;;
}
.fa-location-arrow:before {
  content: &quot;\f124&quot;;
}
.fa-crop:before {
  content: &quot;\f125&quot;;
}
.fa-code-fork:before {
  content: &quot;\f126&quot;;
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: &quot;\f127&quot;;
}
.fa-question:before {
  content: &quot;\f128&quot;;
}
.fa-info:before {
  content: &quot;\f129&quot;;
}
.fa-exclamation:before {
  content: &quot;\f12a&quot;;
}
.fa-superscript:before {
  content: &quot;\f12b&quot;;
}
.fa-subscript:before {
  content: &quot;\f12c&quot;;
}
.fa-eraser:before {
  content: &quot;\f12d&quot;;
}
.fa-puzzle-piece:before {
  content: &quot;\f12e&quot;;
}
.fa-microphone:before {
  content: &quot;\f130&quot;;
}
.fa-microphone-slash:before {
  content: &quot;\f131&quot;;
}
.fa-shield:before {
  content: &quot;\f132&quot;;
}
.fa-calendar-o:before {
  content: &quot;\f133&quot;;
}
.fa-fire-extinguisher:before {
  content: &quot;\f134&quot;;
}
.fa-rocket:before {
  content: &quot;\f135&quot;;
}
.fa-maxcdn:before {
  content: &quot;\f136&quot;;
}
.fa-chevron-circle-left:before {
  content: &quot;\f137&quot;;
}
.fa-chevron-circle-right:before {
  content: &quot;\f138&quot;;
}
.fa-chevron-circle-up:before {
  content: &quot;\f139&quot;;
}
.fa-chevron-circle-down:before {
  content: &quot;\f13a&quot;;
}
.fa-html5:before {
  content: &quot;\f13b&quot;;
}
.fa-css3:before {
  content: &quot;\f13c&quot;;
}
.fa-anchor:before {
  content: &quot;\f13d&quot;;
}
.fa-unlock-alt:before {
  content: &quot;\f13e&quot;;
}
.fa-bullseye:before {
  content: &quot;\f140&quot;;
}
.fa-ellipsis-h:before {
  content: &quot;\f141&quot;;
}
.fa-ellipsis-v:before {
  content: &quot;\f142&quot;;
}
.fa-rss-square:before {
  content: &quot;\f143&quot;;
}
.fa-play-circle:before {
  content: &quot;\f144&quot;;
}
.fa-ticket:before {
  content: &quot;\f145&quot;;
}
.fa-minus-square:before {
  content: &quot;\f146&quot;;
}
.fa-minus-square-o:before {
  content: &quot;\f147&quot;;
}
.fa-level-up:before {
  content: &quot;\f148&quot;;
}
.fa-level-down:before {
  content: &quot;\f149&quot;;
}
.fa-check-square:before {
  content: &quot;\f14a&quot;;
}
.fa-pencil-square:before {
  content: &quot;\f14b&quot;;
}
.fa-external-link-square:before {
  content: &quot;\f14c&quot;;
}
.fa-share-square:before {
  content: &quot;\f14d&quot;;
}
.fa-compass:before {
  content: &quot;\f14e&quot;;
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: &quot;\f150&quot;;
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: &quot;\f151&quot;;
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: &quot;\f152&quot;;
}
.fa-euro:before,
.fa-eur:before {
  content: &quot;\f153&quot;;
}
.fa-gbp:before {
  content: &quot;\f154&quot;;
}
.fa-dollar:before,
.fa-usd:before {
  content: &quot;\f155&quot;;
}
.fa-rupee:before,
.fa-inr:before {
  content: &quot;\f156&quot;;
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: &quot;\f157&quot;;
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: &quot;\f158&quot;;
}
.fa-won:before,
.fa-krw:before {
  content: &quot;\f159&quot;;
}
.fa-bitcoin:before,
.fa-btc:before {
  content: &quot;\f15a&quot;;
}
.fa-file:before {
  content: &quot;\f15b&quot;;
}
.fa-file-text:before {
  content: &quot;\f15c&quot;;
}
.fa-sort-alpha-asc:before {
  content: &quot;\f15d&quot;;
}
.fa-sort-alpha-desc:before {
  content: &quot;\f15e&quot;;
}
.fa-sort-amount-asc:before {
  content: &quot;\f160&quot;;
}
.fa-sort-amount-desc:before {
  content: &quot;\f161&quot;;
}
.fa-sort-numeric-asc:before {
  content: &quot;\f162&quot;;
}
.fa-sort-numeric-desc:before {
  content: &quot;\f163&quot;;
}
.fa-thumbs-up:before {
  content: &quot;\f164&quot;;
}
.fa-thumbs-down:before {
  content: &quot;\f165&quot;;
}
.fa-youtube-square:before {
  content: &quot;\f166&quot;;
}
.fa-youtube:before {
  content: &quot;\f167&quot;;
}
.fa-xing:before {
  content: &quot;\f168&quot;;
}
.fa-xing-square:before {
  content: &quot;\f169&quot;;
}
.fa-youtube-play:before {
  content: &quot;\f16a&quot;;
}
.fa-dropbox:before {
  content: &quot;\f16b&quot;;
}
.fa-stack-overflow:before {
  content: &quot;\f16c&quot;;
}
.fa-instagram:before {
  content: &quot;\f16d&quot;;
}
.fa-flickr:before {
  content: &quot;\f16e&quot;;
}
.fa-adn:before {
  content: &quot;\f170&quot;;
}
.fa-bitbucket:before {
  content: &quot;\f171&quot;;
}
.fa-bitbucket-square:before {
  content: &quot;\f172&quot;;
}
.fa-tumblr:before {
  content: &quot;\f173&quot;;
}
.fa-tumblr-square:before {
  content: &quot;\f174&quot;;
}
.fa-long-arrow-down:before {
  content: &quot;\f175&quot;;
}
.fa-long-arrow-up:before {
  content: &quot;\f176&quot;;
}
.fa-long-arrow-left:before {
  content: &quot;\f177&quot;;
}
.fa-long-arrow-right:before {
  content: &quot;\f178&quot;;
}
.fa-apple:before {
  content: &quot;\f179&quot;;
}
.fa-windows:before {
  content: &quot;\f17a&quot;;
}
.fa-android:before {
  content: &quot;\f17b&quot;;
}
.fa-linux:before {
  content: &quot;\f17c&quot;;
}
.fa-dribbble:before {
  content: &quot;\f17d&quot;;
}
.fa-skype:before {
  content: &quot;\f17e&quot;;
}
.fa-foursquare:before {
  content: &quot;\f180&quot;;
}
.fa-trello:before {
  content: &quot;\f181&quot;;
}
.fa-female:before {
  content: &quot;\f182&quot;;
}
.fa-male:before {
  content: &quot;\f183&quot;;
}
.fa-gittip:before {
  content: &quot;\f184&quot;;
}
.fa-sun-o:before {
  content: &quot;\f185&quot;;
}
.fa-moon-o:before {
  content: &quot;\f186&quot;;
}
.fa-archive:before {
  content: &quot;\f187&quot;;
}
.fa-bug:before {
  content: &quot;\f188&quot;;
}
.fa-vk:before {
  content: &quot;\f189&quot;;
}
.fa-weibo:before {
  content: &quot;\f18a&quot;;
}
.fa-renren:before {
  content: &quot;\f18b&quot;;
}
.fa-pagelines:before {
  content: &quot;\f18c&quot;;
}
.fa-stack-exchange:before {
  content: &quot;\f18d&quot;;
}
.fa-arrow-circle-o-right:before {
  content: &quot;\f18e&quot;;
}
.fa-arrow-circle-o-left:before {
  content: &quot;\f190&quot;;
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: &quot;\f191&quot;;
}
.fa-dot-circle-o:before {
  content: &quot;\f192&quot;;
}
.fa-wheelchair:before {
  content: &quot;\f193&quot;;
}
.fa-vimeo-square:before {
  content: &quot;\f194&quot;;
}
.fa-turkish-lira:before,
.fa-try:before {
  content: &quot;\f195&quot;;
}
.fa-plus-square-o:before {
  content: &quot;\f196&quot;;
}
.fa-space-shuttle:before {
  content: &quot;\f197&quot;;
}
.fa-slack:before {
  content: &quot;\f198&quot;;
}
.fa-envelope-square:before {
  content: &quot;\f199&quot;;
}
.fa-wordpress:before {
  content: &quot;\f19a&quot;;
}
.fa-openid:before {
  content: &quot;\f19b&quot;;
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: &quot;\f19c&quot;;
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: &quot;\f19d&quot;;
}
.fa-yahoo:before {
  content: &quot;\f19e&quot;;
}
.fa-google:before {
  content: &quot;\f1a0&quot;;
}
.fa-reddit:before {
  content: &quot;\f1a1&quot;;
}
.fa-reddit-square:before {
  content: &quot;\f1a2&quot;;
}
.fa-stumbleupon-circle:before {
  content: &quot;\f1a3&quot;;
}
.fa-stumbleupon:before {
  content: &quot;\f1a4&quot;;
}
.fa-delicious:before {
  content: &quot;\f1a5&quot;;
}
.fa-digg:before {
  content: &quot;\f1a6&quot;;
}
.fa-pied-piper:before {
  content: &quot;\f1a7&quot;;
}
.fa-pied-piper-alt:before {
  content: &quot;\f1a8&quot;;
}
.fa-drupal:before {
  content: &quot;\f1a9&quot;;
}
.fa-joomla:before {
  content: &quot;\f1aa&quot;;
}
.fa-language:before {
  content: &quot;\f1ab&quot;;
}
.fa-fax:before {
  content: &quot;\f1ac&quot;;
}
.fa-building:before {
  content: &quot;\f1ad&quot;;
}
.fa-child:before {
  content: &quot;\f1ae&quot;;
}
.fa-paw:before {
  content: &quot;\f1b0&quot;;
}
.fa-spoon:before {
  content: &quot;\f1b1&quot;;
}
.fa-cube:before {
  content: &quot;\f1b2&quot;;
}
.fa-cubes:before {
  content: &quot;\f1b3&quot;;
}
.fa-behance:before {
  content: &quot;\f1b4&quot;;
}
.fa-behance-square:before {
  content: &quot;\f1b5&quot;;
}
.fa-steam:before {
  content: &quot;\f1b6&quot;;
}
.fa-steam-square:before {
  content: &quot;\f1b7&quot;;
}
.fa-recycle:before {
  content: &quot;\f1b8&quot;;
}
.fa-automobile:before,
.fa-car:before {
  content: &quot;\f1b9&quot;;
}
.fa-cab:before,
.fa-taxi:before {
  content: &quot;\f1ba&quot;;
}
.fa-tree:before {
  content: &quot;\f1bb&quot;;
}
.fa-spotify:before {
  content: &quot;\f1bc&quot;;
}
.fa-deviantart:before {
  content: &quot;\f1bd&quot;;
}
.fa-soundcloud:before {
  content: &quot;\f1be&quot;;
}
.fa-database:before {
  content: &quot;\f1c0&quot;;
}
.fa-file-pdf-o:before {
  content: &quot;\f1c1&quot;;
}
.fa-file-word-o:before {
  content: &quot;\f1c2&quot;;
}
.fa-file-excel-o:before {
  content: &quot;\f1c3&quot;;
}
.fa-file-powerpoint-o:before {
  content: &quot;\f1c4&quot;;
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: &quot;\f1c5&quot;;
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: &quot;\f1c6&quot;;
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: &quot;\f1c7&quot;;
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: &quot;\f1c8&quot;;
}
.fa-file-code-o:before {
  content: &quot;\f1c9&quot;;
}
.fa-vine:before {
  content: &quot;\f1ca&quot;;
}
.fa-codepen:before {
  content: &quot;\f1cb&quot;;
}
.fa-jsfiddle:before {
  content: &quot;\f1cc&quot;;
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: &quot;\f1cd&quot;;
}
.fa-circle-o-notch:before {
  content: &quot;\f1ce&quot;;
}
.fa-ra:before,
.fa-rebel:before {
  content: &quot;\f1d0&quot;;
}
.fa-ge:before,
.fa-empire:before {
  content: &quot;\f1d1&quot;;
}
.fa-git-square:before {
  content: &quot;\f1d2&quot;;
}
.fa-git:before {
  content: &quot;\f1d3&quot;;
}
.fa-hacker-news:before {
  content: &quot;\f1d4&quot;;
}
.fa-tencent-weibo:before {
  content: &quot;\f1d5&quot;;
}
.fa-qq:before {
  content: &quot;\f1d6&quot;;
}
.fa-wechat:before,
.fa-weixin:before {
  content: &quot;\f1d7&quot;;
}
.fa-send:before,
.fa-paper-plane:before {
  content: &quot;\f1d8&quot;;
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: &quot;\f1d9&quot;;
}
.fa-history:before {
  content: &quot;\f1da&quot;;
}
.fa-circle-thin:before {
  content: &quot;\f1db&quot;;
}
.fa-header:before {
  content: &quot;\f1dc&quot;;
}
.fa-paragraph:before {
  content: &quot;\f1dd&quot;;
}
.fa-sliders:before {
  content: &quot;\f1de&quot;;
}
.fa-share-alt:before {
  content: &quot;\f1e0&quot;;
}
.fa-share-alt-square:before {
  content: &quot;\f1e1&quot;;
}
.fa-bomb:before {
  content: &quot;\f1e2&quot;;
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: &quot;\f1e3&quot;;
}
.fa-tty:before {
  content: &quot;\f1e4&quot;;
}
.fa-binoculars:before {
  content: &quot;\f1e5&quot;;
}
.fa-plug:before {
  content: &quot;\f1e6&quot;;
}
.fa-slideshare:before {
  content: &quot;\f1e7&quot;;
}
.fa-twitch:before {
  content: &quot;\f1e8&quot;;
}
.fa-yelp:before {
  content: &quot;\f1e9&quot;;
}
.fa-newspaper-o:before {
  content: &quot;\f1ea&quot;;
}
.fa-wifi:before {
  content: &quot;\f1eb&quot;;
}
.fa-calculator:before {
  content: &quot;\f1ec&quot;;
}
.fa-paypal:before {
  content: &quot;\f1ed&quot;;
}
.fa-google-wallet:before {
  content: &quot;\f1ee&quot;;
}
.fa-cc-visa:before {
  content: &quot;\f1f0&quot;;
}
.fa-cc-mastercard:before {
  content: &quot;\f1f1&quot;;
}
.fa-cc-discover:before {
  content: &quot;\f1f2&quot;;
}
.fa-cc-amex:before {
  content: &quot;\f1f3&quot;;
}
.fa-cc-paypal:before {
  content: &quot;\f1f4&quot;;
}
.fa-cc-stripe:before {
  content: &quot;\f1f5&quot;;
}
.fa-bell-slash:before {
  content: &quot;\f1f6&quot;;
}
.fa-bell-slash-o:before {
  content: &quot;\f1f7&quot;;
}
.fa-trash:before {
  content: &quot;\f1f8&quot;;
}
.fa-copyright:before {
  content: &quot;\f1f9&quot;;
}
.fa-at:before {
  content: &quot;\f1fa&quot;;
}
.fa-eyedropper:before {
  content: &quot;\f1fb&quot;;
}
.fa-paint-brush:before {
  content: &quot;\f1fc&quot;;
}
.fa-birthday-cake:before {
  content: &quot;\f1fd&quot;;
}
.fa-area-chart:before {
  content: &quot;\f1fe&quot;;
}
.fa-pie-chart:before {
  content: &quot;\f200&quot;;
}
.fa-line-chart:before {
  content: &quot;\f201&quot;;
}
.fa-lastfm:before {
  content: &quot;\f202&quot;;
}
.fa-lastfm-square:before {
  content: &quot;\f203&quot;;
}
.fa-toggle-off:before {
  content: &quot;\f204&quot;;
}
.fa-toggle-on:before {
  content: &quot;\f205&quot;;
}
.fa-bicycle:before {
  content: &quot;\f206&quot;;
}
.fa-bus:before {
  content: &quot;\f207&quot;;
}
.fa-ioxhost:before {
  content: &quot;\f208&quot;;
}
.fa-angellist:before {
  content: &quot;\f209&quot;;
}
.fa-cc:before {
  content: &quot;\f20a&quot;;
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: &quot;\f20b&quot;;
}
.fa-meanpath:before {
  content: &quot;\f20c&quot;;
}
/<em>!
</em>
<em> IPython base
</em>
<em>/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/</em> Make the page background atleast 100% the height of the view port <em>/
/</em> Make the page itself atleast 70% the height of the view port <em>/
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/</em> Flexible box model classes <em>/
/</em> Taken from Alex Russell <a href="http://infrequently.org/2009/08/css-3-progress/">http://infrequently.org/2009/08/css-3-progress/</a> <em>/
/</em> This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
<code>Modern browsers</code> comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
<code>Old browsers</code> comments below.
<em>/
.hbox {
  /</em> Old browsers <em>/
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /</em> Modern browsers <em>/
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox &gt; </em> {
  /<em> Old browsers </em>/
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /<em> Modern browsers </em>/
  flex: none;
}
.vbox {
  /<em> Old browsers </em>/
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /<em> Modern browsers </em>/
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox &gt; <em> {
  /</em> Old browsers <em>/
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /</em> Modern browsers <em>/
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /</em> Old browsers <em>/
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /</em> Modern browsers <em>/
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /</em> Old browsers <em>/
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /</em> Modern browsers <em>/
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /</em> Old browsers <em>/
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /</em> Modern browsers <em>/
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /</em> Old browsers <em>/
  /</em> Old browsers <em>/
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /</em> Modern browsers <em>/
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /</em> Old browsers <em>/
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /</em> Modern browsers <em>/
  flex: 2;
}
.box-group1 {
  /</em>  Deprecated <em>/
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /</em> Deprecated <em>/
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /</em> Old browsers <em>/
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /</em> Modern browsers <em>/
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /</em> Old browsers <em>/
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /</em> Modern browsers <em>/
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /</em> Old browsers <em>/
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /</em> Modern browsers <em>/
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /</em> Old browsers <em>/
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /</em> Modern browsers <em>/
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /</em> Old browsers <em>/
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /</em> Modern browsers <em>/
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /</em> Old browsers <em>/
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /</em> Modern browsers <em>/
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /</em> Old browsers <em>/
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /</em> Modern browsers <em>/
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /</em> Old browsers <em>/
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /</em> Modern browsers <em>/
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /</em> Old browsers <em>/
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /</em> Modern browsers <em>/
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /</em> Old browsers <em>/
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /</em> Modern browsers <em>/
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error &gt; h1 {
  font-size: 500%;
  line-height: normal;
}
div.error &gt; p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
/**
 </em> Primary styles
 <em>
 </em> Author: Jupyter Development Team
 <em>/
body {
  background-color: #fff;
  /</em> This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below <em>/
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body &gt; #header {
  /</em> Initially hidden to prevent FLOUC <em>/
  display: none;
  background-color: #fff;
  /</em> Display over codemirror <em>/
  position: relative;
  z-index: 100;
}
body &gt; #header #header-container {
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body &gt; #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body &gt; #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
@media (max-width: 991px) {
  #ipython_notebook {
    margin-left: 10px;
  }
}
[dir=&quot;rtl&quot;] #ipython_notebook {
  float: right !important;
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/</em> Smaller buttons <em>/
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#login_widget {
  float: right;
}
span#login_widget &gt; .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget &gt; .button:focus,
#logout:focus,
span#login_widget &gt; .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget &gt; .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget &gt; .button:active,
#logout:active,
span#login_widget &gt; .button.active,
#logout.active,
.open &gt; .dropdown-togglespan#login_widget &gt; .button,
.open &gt; .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget &gt; .button:active:hover,
#logout:active:hover,
span#login_widget &gt; .button.active:hover,
#logout.active:hover,
.open &gt; .dropdown-togglespan#login_widget &gt; .button:hover,
.open &gt; .dropdown-toggle#logout:hover,
span#login_widget &gt; .button:active:focus,
#logout:active:focus,
span#login_widget &gt; .button.active:focus,
#logout.active:focus,
.open &gt; .dropdown-togglespan#login_widget &gt; .button:focus,
.open &gt; .dropdown-toggle#logout:focus,
span#login_widget &gt; .button:active.focus,
#logout:active.focus,
span#login_widget &gt; .button.active.focus,
#logout.active.focus,
.open &gt; .dropdown-togglespan#login_widget &gt; .button.focus,
.open &gt; .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget &gt; .button:active,
#logout:active,
span#login_widget &gt; .button.active,
#logout.active,
.open &gt; .dropdown-togglespan#login_widget &gt; .button,
.open &gt; .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget &gt; .button.disabled:hover,
#logout.disabled:hover,
span#login_widget &gt; .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget &gt; .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget &gt; .button.disabled:focus,
#logout.disabled:focus,
span#login_widget &gt; .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget &gt; .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget &gt; .button.disabled.focus,
#logout.disabled.focus,
span#login_widget &gt; .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget &gt; .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget &gt; .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header &gt; span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /</em> Old browsers <em>/
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /</em> Modern browsers <em>/
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/</em>!
<em>
</em> IPython auth
<em>
</em>/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
/<em>!
</em>
<em> IPython tree view
</em>
<em>/
/</em> We need an invisible input field on top of the sentense<em>/
/</em> &quot;Drag file onto the list ...&quot; <em>/
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  text-align: center;
  vertical-align: middle;
  display: inline;
  opacity: 0;
  z-index: 2;
  width: 12ex;
  margin-right: -12ex;
}
.alternate_upload .btn-upload {
  height: 22px;
}
/**
 </em> Primary styles
 <em>
 </em> Author: Jupyter Development Team
 <em>/
[dir=&quot;rtl&quot;] #tabs li {
  float: right;
}
ul#tabs {
  margin-bottom: 4px;
}
[dir=&quot;rtl&quot;] ul#tabs {
  margin-right: 0px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
[dir=&quot;rtl&quot;] .list_toolbar .tree-buttons {
  float: left !important;
}
[dir=&quot;rtl&quot;] .list_toolbar .pull-right {
  padding-top: 1px;
  float: left !important;
}
[dir=&quot;rtl&quot;] .list_toolbar .pull-left {
  float: right !important;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class</em>=&quot;span&quot;] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container &gt; div {
  border-bottom: 1px solid #ddd;
}
.list_container &gt; div:hover .list-item {
  background-color: red;
}
.list_container &gt; div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header &gt; div,
.list_item &gt; div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header &gt; div input,
.list_item &gt; div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: baseline;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header &gt; div .item_link,
.list_item &gt; div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons &gt; .btn,
.item_buttons &gt; .btn-group,
.item_buttons &gt; .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name &gt; .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
#tree-selector {
  padding-right: 0px;
}
[dir=&quot;rtl&quot;] #tree-selector a {
  float: right;
}
#button-select-all {
  min-width: 50px;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: &quot;\f114&quot;;
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: &quot;\f02d&quot;;
  position: relative;
  top: -1px;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: &quot;\f02d&quot;;
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: &quot;\f016&quot;;
  position: relative;
  top: -2px;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
[dir=&quot;rtl&quot;] #new-menu {
  text-align: right;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: &quot;\f096&quot;;
}
.kernel-menu-icon:before {
  content: &quot;\f096&quot;;
}
.kernel-menu-icon-current:before {
  content: &quot;\f00c&quot;;
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
[dir=&quot;rtl&quot;] #running .col-sm-8 {
  float: right !important;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/<em>!
</em>
<em> IPython text editor webapp
</em>
<em>/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: &quot;\f00c&quot;;
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /</em> Use a negative 1 bottom margin, so the border overlaps the border of the
    header <em>/
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: &quot;\f00c&quot;;
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
/</em>!
<em>
</em> IPython notebook
<em>
</em>/
/<em> CSS font colors for translated ANSI colors. </em>/
.ansibold {
  font-weight: bold;
}
/<em> use dark versions for foreground, to improve visibility </em>/
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/<em> and light for background, for the same reason </em>/
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /<em> Old browsers </em>/
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /<em> Modern browsers </em>/
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /<em> This acts as a spacer between cells, that is outside the border </em>/
  margin: 0px;
  outline: none;
  border-left-width: 1px;
  padding-left: 5px;
  background: linear-gradient(to right, transparent -40px, transparent 1px, transparent 1px, transparent 100%);
}
div.cell.jupyter-soft-selected {
  border-left-color: #90CAF9;
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected {
  border-color: #ababab;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 5px, transparent 5px, transparent 100%);
}
@media print {
  div.cell.selected {
    border-color: transparent;
  }
}
div.cell.selected.jupyter-soft-selected {
  border-left-width: 0;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 7px, #E3F2FD 7px, #E3F2FD 100%);
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #66BB6A -40px, #66BB6A 5px, transparent 5px, transparent 100%);
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /<em> This needs to be wide enough for 3 digit prompt numbers: In[100]: </em>/
  min-width: 14ex;
  /<em> This padding is tuned to match the padding on the CodeMirror editor. </em>/
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /<em> This has to match that of the the CodeMirror class line-height below </em>/
  line-height: 1.21429em;
  /<em> Don&#39;t highlight prompt number selection </em>/
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /<em> Use default cursor </em>/
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  min-width: 0;
  /<em> Old browsers </em>/
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /<em> Modern browsers </em>/
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /<em> Old browsers </em>/
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /<em> Modern browsers </em>/
  flex: 1;
}
/<em> input_area and input_prompt must match in top border and margin for alignment </em>/
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/<em> This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. </em>/
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /<em> Old browsers </em>/
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /<em> Modern browsers </em>/
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell &gt; div.prompt {
    display: none;
  }
}
div.code_cell {
  /<em> avoid page breaking on code cells when printing </em>/
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/<em> any special styling for code cells that are currently running goes here </em>/
div.input {
  page-break-inside: avoid;
  /<em> Old browsers </em>/
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /<em> Modern browsers </em>/
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /<em> Old browsers </em>/
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /<em> Modern browsers </em>/
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/<em> input_area and input_prompt must match in top border and margin for alignment </em>/
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area &gt; div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area &gt; div.highlight &gt; pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/<em> The following gets added to the <head> if it is detected that the user has a
 </em> monospace font with inconsistent normal/bold/italic height.  See
 <em> notebookmain.js.  Such fonts will have keywords vertically offset with
 </em> respect to the rest of the text.  The user should select a better font.
 <em> See: <a href="https://github.com/ipython/ipython/issues/1503">https://github.com/ipython/ipython/issues/1503</a>
 </em>
 <em> .CodeMirror span {
 </em>      vertical-align: bottom;
 <em> }
 </em>/
.CodeMirror {
  line-height: 1.21429em;
  /<em> Changed from 1em to our global default </em>/
  font-size: 14px;
  height: auto;
  /<em> Changed to auto to autogrow </em>/
  background: none;
  /<em> Changed from white to allow our bg to show through </em>/
}
.CodeMirror-scroll {
  /<em>  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.</em>/
  /<em>  We have found that if it is visible, vertical scrollbars appear with font size changes.</em>/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /<em> In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because </em>/
  /<em> we have set a different line-height and want this to scale with that. </em>/
  padding: 0.4em;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /<em> In CM3 this went to 4px from 0 in CM2. We need the 0 value because of how we size </em>/
  /<em> .CodeMirror-lines </em>/
  padding: 0;
  border: 0;
  border-radius: 0;
}
/<em>

Original style from softwaremaniacs.org (c) Ivan Sagalaev <a href="&#x6d;&#x61;&#105;&#108;&#x74;&#111;&#58;&#77;&#97;&#x6e;&#x69;&#x61;&#x63;&#64;&#83;&#111;&#x66;&#116;&#x77;&#97;&#114;&#x65;&#77;&#x61;&#x6e;&#x69;&#97;&#x63;&#x73;&#x2e;&#x4f;&#x72;&#103;">&#77;&#97;&#x6e;&#x69;&#x61;&#x63;&#64;&#83;&#111;&#x66;&#116;&#x77;&#97;&#114;&#x65;&#77;&#x61;&#x6e;&#x69;&#97;&#x63;&#x73;&#x2e;&#x4f;&#x72;&#103;</a>
Adapted from GitHub theme

</em>/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/<em> previously not defined, copying from default codemirror </em>/
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/<em> apply the same style to codemirror </em>/
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /<em> this position must be relative to enable descendents to be absolute within it </em>/
  position: relative;
  /<em> Old browsers </em>/
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /<em> Modern browsers </em>/
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/<em> class for the output area when it should be height-limited </em>/
div.output_scroll {
  /<em> ideally, this would be max-height, but FF barfs all over that </em>/
  height: 24em;
  /<em> FF needs this </em>and the wrapper<em> to specify full width, or it will shrinkwrap </em>/
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/<em> output div while it is collapsed </em>/
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /<em> Old browsers </em>/
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /<em> Modern browsers </em>/
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /<em> use inner shadow to get border that is computed the same on WebKit/FF </em>/
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/<em> This class is the outer container of all output sections. </em>/
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /<em> Old browsers </em>/
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /<em> Modern browsers </em>/
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
/<em> This is needed to protect the pre formating from global settings such
   as that of bootstrap </em>/
.output {
  /<em> Old browsers </em>/
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /<em> Modern browsers </em>/
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /<em> Old browsers </em>/
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /<em> Modern browsers </em>/
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/<em> This class is for the output subarea inside the output_area and after
   the prompt div. </em>/
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /<em> Old browsers </em>/
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /<em> Modern browsers </em>/
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output<em>subarea {
  overflow-x: visible;
}
/* The rest of the output</em><em> classes are for special styling of the different
   output types </em>/
/<em> all text output has this class: </em>/
div.output_text {
  text-align: left;
  color: #000;
  /<em> This has to match that of the the CodeMirror class line-height below </em>/
  line-height: 1.21429em;
}
/<em> stdout/stderr are &#39;text&#39; as well as &#39;stream&#39;, but execute_result/error are </em>not<em> streams </em>/
div.output_stderr {
  background: #fdd;
  /<em> very light red background for stderr </em>/
}
div.output_latex {
  text-align: left;
}
/<em> Empty output_javascript divs should have no height </em>/
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/<em> raw_input styles </em>/
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /<em> nothing needed here. </em>/
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /<em> make sure input baseline aligns with prompt </em>/
  vertical-align: baseline;
  /<em> padding + margin = 0.5em between prompt and cursor </em>/
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /<em> any extras will just be numbers: </em>/
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul {
  list-style: disc;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ul ul {
  list-style: square;
  margin: 0em 2em;
}
.rendered_html ul ul ul {
  list-style: circle;
  margin: 0em 2em;
}
.rendered_html ol {
  list-style: decimal;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
  margin: 0em 2em;
}
.rendered_html <em> + ul {
  margin-top: 1em;
}
.rendered_html </em> + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  background-color: #fff;
  color: #000;
  font-size: 100%;
  padding: 0px;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: 1px solid black;
  border-collapse: collapse;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  border: 1px solid black;
  border-collapse: collapse;
  margin: 1em 2em;
}
.rendered_html td,
.rendered_html th {
  text-align: left;
  vertical-align: middle;
  padding: 4px;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html <em> + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html </em> + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html <em> + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
div.text_cell {
  /</em> Old browsers <em>/
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /</em> Modern browsers <em>/
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell &gt; div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /</em>font-family: &quot;Helvetica Neue&quot;, Arial, Helvetica, Geneva, sans-serif;<em>/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: &quot;Helvetica Neue&quot;, Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/</em>!
<em>
</em> IPython notebook webapp
<em>
</em>/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /<em> This spaces the page away from the edge of the notebook area </em>/
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/<em> Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 </em>/
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app &gt; #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/<em> CSS for the cell toolbar </em>/
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /<em> Old browsers </em>/
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /<em> Modern browsers </em>/
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /<em> Old browsers </em>/
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /<em> Modern browsers </em>/
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/<em> ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
</em>/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget {
  float: right !important;
  float: right;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu &gt; .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
.dropdown-submenu:hover &gt; .dropdown-menu {
  display: block;
}
.dropdown-submenu &gt; a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: &quot;\f0da&quot;;
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu &gt; a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu &gt; a:after.pull-right {
  margin-left: .3em;
}
.dropdown-submenu:hover &gt; a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left &gt; .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: &quot;\f040&quot;;
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: &#39; &#39;;
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: &quot;\f10c&quot;;
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: &quot;\f111&quot;;
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: &quot;\f1e2&quot;;
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: &quot;\f127&quot;;
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open &gt; .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open &gt; .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open &gt; .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open &gt; .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open &gt; .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open &gt; .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open &gt; .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open &gt; .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open &gt; .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open &gt; .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open &gt; .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open &gt; .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open &gt; .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open &gt; .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open &gt; .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open &gt; .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open &gt; .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open &gt; .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open &gt; .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open &gt; .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open &gt; .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open &gt; .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open &gt; .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open &gt; .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open &gt; .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /<em> Display over codemirror </em>/
  z-index: 100;
  /<em> Hack which prevents jquery ui resizable from changing top. </em>/
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /<em> This injects handle bars (a short, wide = symbol) for 
        the resize handle. </em>/
}
div#pager .ui-resizable-handle::after {
  content: &#39;&#39;;
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /<em> Old browsers </em>/
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /<em> Modern browsers </em>/
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 21ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /<em> Old browsers </em>/
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /<em> Modern browsers </em>/
  flex: 1;
}
span.save_widget {
  margin-top: 6px;
}
span.save_widget span.filename {
  height: 1em;
  line-height: 1em;
  padding: 3px;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
.pulse,
.dropdown-menu &gt; li &gt; a.pulse,
li.pulse &gt; a.dropdown-toggle,
li.pulse.open &gt; a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/<strong>
 <em> Primary styles
 </em>
 <em> Author: Jupyter Development Team
 </em>/
/</strong> WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 <em> of chance of beeing generated from the ../less/[samename].less file, you can
 </em> try to get back the less file by reverting somme commit in history
 <strong>/
/<em>
 </em> We&#39;ll try to get something pretty, so we
 <em> have some strange css to have the scroll bar on
 </em> the left with fix button on the top right of the tooltip
 <em>/
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/</em>properties of tooltip after &quot;expand&quot;<em>/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/</em>properties of tooltip before &quot;expand&quot;<em>/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /</em>avoid the button to overlap on some docstring<em>/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /</em>fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: &quot;&quot;;
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list &gt; li &gt; a {
  /</strong> Firefox bug <strong>/
  /<em> see <a href="https://github.com/jupyter/notebook/issues/559">https://github.com/jupyter/notebook/issues/559</a> </em>/
  white-space: normal;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  display: none;
}
.command-shortcut:before {
  content: &quot;(command)&quot;;
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: &quot;(edit)&quot;;
  padding-right: 3px;
  color: #777777;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  width: 100%;
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal .xterm-rows {
  padding: 10px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/<em># sourceMappingURL=style.min.css.map </em>/
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /<em> Comment </em>/
.highlight .err { border: 1px solid #FF0000 } /<em> Error </em>/
.highlight .k { color: #008000; font-weight: bold } /<em> Keyword </em>/
.highlight .o { color: #666666 } /<em> Operator </em>/
.highlight .ch { color: #408080; font-style: italic } /<em> Comment.Hashbang </em>/
.highlight .cm { color: #408080; font-style: italic } /<em> Comment.Multiline </em>/
.highlight .cp { color: #BC7A00 } /<em> Comment.Preproc </em>/
.highlight .cpf { color: #408080; font-style: italic } /<em> Comment.PreprocFile </em>/
.highlight .c1 { color: #408080; font-style: italic } /<em> Comment.Single </em>/
.highlight .cs { color: #408080; font-style: italic } /<em> Comment.Special </em>/
.highlight .gd { color: #A00000 } /<em> Generic.Deleted </em>/
.highlight .ge { font-style: italic } /<em> Generic.Emph </em>/
.highlight .gr { color: #FF0000 } /<em> Generic.Error </em>/
.highlight .gh { color: #000080; font-weight: bold } /<em> Generic.Heading </em>/
.highlight .gi { color: #00A000 } /<em> Generic.Inserted </em>/
.highlight .go { color: #888888 } /<em> Generic.Output </em>/
.highlight .gp { color: #000080; font-weight: bold } /<em> Generic.Prompt </em>/
.highlight .gs { font-weight: bold } /<em> Generic.Strong </em>/
.highlight .gu { color: #800080; font-weight: bold } /<em> Generic.Subheading </em>/
.highlight .gt { color: #0044DD } /<em> Generic.Traceback </em>/
.highlight .kc { color: #008000; font-weight: bold } /<em> Keyword.Constant </em>/
.highlight .kd { color: #008000; font-weight: bold } /<em> Keyword.Declaration </em>/
.highlight .kn { color: #008000; font-weight: bold } /<em> Keyword.Namespace </em>/
.highlight .kp { color: #008000 } /<em> Keyword.Pseudo </em>/
.highlight .kr { color: #008000; font-weight: bold } /<em> Keyword.Reserved </em>/
.highlight .kt { color: #B00040 } /<em> Keyword.Type </em>/
.highlight .m { color: #666666 } /<em> Literal.Number </em>/
.highlight .s { color: #BA2121 } /<em> Literal.String </em>/
.highlight .na { color: #7D9029 } /<em> Name.Attribute </em>/
.highlight .nb { color: #008000 } /<em> Name.Builtin </em>/
.highlight .nc { color: #0000FF; font-weight: bold } /<em> Name.Class </em>/
.highlight .no { color: #880000 } /<em> Name.Constant </em>/
.highlight .nd { color: #AA22FF } /<em> Name.Decorator </em>/
.highlight .ni { color: #999999; font-weight: bold } /<em> Name.Entity </em>/
.highlight .ne { color: #D2413A; font-weight: bold } /<em> Name.Exception </em>/
.highlight .nf { color: #0000FF } /<em> Name.Function </em>/
.highlight .nl { color: #A0A000 } /<em> Name.Label </em>/
.highlight .nn { color: #0000FF; font-weight: bold } /<em> Name.Namespace </em>/
.highlight .nt { color: #008000; font-weight: bold } /<em> Name.Tag </em>/
.highlight .nv { color: #19177C } /<em> Name.Variable </em>/
.highlight .ow { color: #AA22FF; font-weight: bold } /<em> Operator.Word </em>/
.highlight .w { color: #bbbbbb } /<em> Text.Whitespace </em>/
.highlight .mb { color: #666666 } /<em> Literal.Number.Bin </em>/
.highlight .mf { color: #666666 } /<em> Literal.Number.Float </em>/
.highlight .mh { color: #666666 } /<em> Literal.Number.Hex </em>/
.highlight .mi { color: #666666 } /<em> Literal.Number.Integer </em>/
.highlight .mo { color: #666666 } /<em> Literal.Number.Oct </em>/
.highlight .sa { color: #BA2121 } /<em> Literal.String.Affix </em>/
.highlight .sb { color: #BA2121 } /<em> Literal.String.Backtick </em>/
.highlight .sc { color: #BA2121 } /<em> Literal.String.Char </em>/
.highlight .dl { color: #BA2121 } /<em> Literal.String.Delimiter </em>/
.highlight .sd { color: #BA2121; font-style: italic } /<em> Literal.String.Doc </em>/
.highlight .s2 { color: #BA2121 } /<em> Literal.String.Double </em>/
.highlight .se { color: #BB6622; font-weight: bold } /<em> Literal.String.Escape </em>/
.highlight .sh { color: #BA2121 } /<em> Literal.String.Heredoc </em>/
.highlight .si { color: #BB6688; font-weight: bold } /<em> Literal.String.Interpol </em>/
.highlight .sx { color: #008000 } /<em> Literal.String.Other </em>/
.highlight .sr { color: #BB6688 } /<em> Literal.String.Regex </em>/
.highlight .s1 { color: #BA2121 } /<em> Literal.String.Single </em>/
.highlight .ss { color: #19177C } /<em> Literal.String.Symbol </em>/
.highlight .bp { color: #008000 } /<em> Name.Builtin.Pseudo </em>/
.highlight .fm { color: #0000FF } /<em> Name.Function.Magic </em>/
.highlight .vc { color: #19177C } /<em> Name.Variable.Class </em>/
.highlight .vg { color: #19177C } /<em> Name.Variable.Global </em>/
.highlight .vi { color: #19177C } /<em> Name.Variable.Instance </em>/
.highlight .vm { color: #19177C } /<em> Name.Variable.Magic </em>/
.highlight .il { color: #666666 } /<em> Literal.Number.Integer.Long </em>/
    </style>
<style type="text/css">

/<em> Temporary definitions which will become obsolete with Notebook release 5.0 </em>/
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }

    </style>


<style type="text/css">
/<em> Overrides of notebook CSS for static HTML export </em>/
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}

@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
<!-- Load mathjax -->
    
    <!-- MathJax configuration -->
    
    <!-- End of mathjax configuration --></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[1]:</div>


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>3SsnPorch</th>
      <th>BedroomAbvGr</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>BsmtUnfSF</th>
      <th>CentralAir</th>
      <th>...</th>
      <th>SaleType_CWD</th>
      <th>SaleType_Con</th>
      <th>SaleType_ConLD</th>
      <th>SaleType_ConLI</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>Street_Pave</th>
      <th>Utilities_NoSeWa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>9</td>
      <td>706.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>150.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>978.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>284.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>9</td>
      <td>486.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>434.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>9</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>540.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>16</td>
      <td>655.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>490.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 242 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[2]:</div>


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>3SsnPorch</th>
      <th>BedroomAbvGr</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>BsmtUnfSF</th>
      <th>CentralAir</th>
      <th>...</th>
      <th>SaleType_CWD</th>
      <th>SaleType_Con</th>
      <th>SaleType_ConLD</th>
      <th>SaleType_ConLI</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>Street_Pave</th>
      <th>Utilities_NoSeWa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1stFlrSF</th>
      <td>1.000000</td>
      <td>-0.202646</td>
      <td>0.056104</td>
      <td>0.136324</td>
      <td>0.445863</td>
      <td>0.097117</td>
      <td>0.238363</td>
      <td>0.002471</td>
      <td>0.317987</td>
      <td>0.146953</td>
      <td>...</td>
      <td>0.033381</td>
      <td>0.007559</td>
      <td>-0.011789</td>
      <td>0.006094</td>
      <td>-0.043721</td>
      <td>0.221219</td>
      <td>-0.008215</td>
      <td>-0.198056</td>
      <td>0.005950</td>
      <td>0.012287</td>
    </tr>
    <tr>
      <th>2ndFlrSF</th>
      <td>-0.202646</td>
      <td>1.000000</td>
      <td>-0.024358</td>
      <td>0.497711</td>
      <td>-0.137079</td>
      <td>-0.099260</td>
      <td>-0.152625</td>
      <td>-0.029120</td>
      <td>0.004469</td>
      <td>-0.011803</td>
      <td>...</td>
      <td>0.007628</td>
      <td>0.003778</td>
      <td>-0.018808</td>
      <td>0.016175</td>
      <td>0.012602</td>
      <td>0.010810</td>
      <td>-0.036082</td>
      <td>0.026769</td>
      <td>0.046983</td>
      <td>-0.020818</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>0.056104</td>
      <td>-0.024358</td>
      <td>1.000000</td>
      <td>-0.026945</td>
      <td>0.026451</td>
      <td>-0.029993</td>
      <td>-0.004511</td>
      <td>0.029822</td>
      <td>0.020764</td>
      <td>0.030692</td>
      <td>...</td>
      <td>-0.006098</td>
      <td>-0.004309</td>
      <td>-0.009162</td>
      <td>-0.006820</td>
      <td>-0.006820</td>
      <td>0.019596</td>
      <td>-0.005279</td>
      <td>-0.014211</td>
      <td>0.007473</td>
      <td>-0.003046</td>
    </tr>
    <tr>
      <th>BedroomAbvGr</th>
      <td>0.136324</td>
      <td>0.497711</td>
      <td>-0.026945</td>
      <td>1.000000</td>
      <td>-0.087304</td>
      <td>-0.019908</td>
      <td>-0.111115</td>
      <td>0.045787</td>
      <td>0.149744</td>
      <td>-0.021231</td>
      <td>...</td>
      <td>0.030047</td>
      <td>-0.028730</td>
      <td>-0.013874</td>
      <td>0.081051</td>
      <td>-0.029071</td>
      <td>-0.054679</td>
      <td>0.022234</td>
      <td>0.046933</td>
      <td>0.030573</td>
      <td>0.000620</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>0.445863</td>
      <td>-0.137079</td>
      <td>0.026451</td>
      <td>-0.087304</td>
      <td>1.000000</td>
      <td>-0.050117</td>
      <td>0.611558</td>
      <td>0.060164</td>
      <td>-0.495251</td>
      <td>0.166468</td>
      <td>...</td>
      <td>0.008951</td>
      <td>0.030694</td>
      <td>-0.021376</td>
      <td>0.022726</td>
      <td>-0.017825</td>
      <td>0.044883</td>
      <td>0.010652</td>
      <td>-0.024778</td>
      <td>-0.015643</td>
      <td>-0.019100</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>0.097117</td>
      <td>-0.099260</td>
      <td>-0.029993</td>
      <td>-0.019908</td>
      <td>-0.050117</td>
      <td>1.000000</td>
      <td>0.128686</td>
      <td>0.059713</td>
      <td>-0.209294</td>
      <td>0.039936</td>
      <td>...</td>
      <td>0.076364</td>
      <td>-0.010691</td>
      <td>-0.022733</td>
      <td>-0.016921</td>
      <td>0.035715</td>
      <td>-0.087162</td>
      <td>-0.013098</td>
      <td>0.036178</td>
      <td>-0.038487</td>
      <td>0.049913</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>0.238363</td>
      <td>-0.152625</td>
      <td>-0.004511</td>
      <td>-0.111115</td>
      <td>0.611558</td>
      <td>0.128686</td>
      <td>1.000000</td>
      <td>-0.119617</td>
      <td>-0.380957</td>
      <td>0.101755</td>
      <td>...</td>
      <td>-0.016173</td>
      <td>0.088571</td>
      <td>-0.014176</td>
      <td>-0.004522</td>
      <td>-0.004522</td>
      <td>-0.014893</td>
      <td>0.061841</td>
      <td>0.015132</td>
      <td>-0.071029</td>
      <td>-0.018176</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>0.002471</td>
      <td>-0.029120</td>
      <td>0.029822</td>
      <td>0.045787</td>
      <td>0.060164</td>
      <td>0.059713</td>
      <td>-0.119617</td>
      <td>1.000000</td>
      <td>-0.083167</td>
      <td>0.038275</td>
      <td>...</td>
      <td>0.036786</td>
      <td>-0.008258</td>
      <td>-0.017560</td>
      <td>-0.013070</td>
      <td>0.030299</td>
      <td>-0.021547</td>
      <td>-0.010117</td>
      <td>0.012215</td>
      <td>0.014323</td>
      <td>0.091007</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>0.317987</td>
      <td>0.004469</td>
      <td>0.020764</td>
      <td>0.149744</td>
      <td>-0.495251</td>
      <td>-0.209294</td>
      <td>-0.380957</td>
      <td>-0.083167</td>
      <td>1.000000</td>
      <td>0.020060</td>
      <td>...</td>
      <td>-0.028685</td>
      <td>-0.012681</td>
      <td>-0.000835</td>
      <td>0.001853</td>
      <td>-0.033900</td>
      <td>0.249236</td>
      <td>-0.002593</td>
      <td>-0.198960</td>
      <td>0.035229</td>
      <td>-0.012639</td>
    </tr>
    <tr>
      <th>CentralAir</th>
      <td>0.146953</td>
      <td>-0.011803</td>
      <td>0.030692</td>
      <td>-0.021231</td>
      <td>0.166468</td>
      <td>0.039936</td>
      <td>0.101755</td>
      <td>0.038275</td>
      <td>0.020060</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.039299</td>
      <td>0.009771</td>
      <td>-0.085660</td>
      <td>0.015465</td>
      <td>-0.079604</td>
      <td>0.079661</td>
      <td>0.011971</td>
      <td>-0.037373</td>
      <td>0.069869</td>
      <td>0.006907</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>-0.065292</td>
      <td>0.061989</td>
      <td>-0.037305</td>
      <td>0.034737</td>
      <td>-0.102303</td>
      <td>0.036543</td>
      <td>-0.053614</td>
      <td>-0.011195</td>
      <td>-0.002538</td>
      <td>-0.156913</td>
      <td>...</td>
      <td>-0.018834</td>
      <td>-0.013308</td>
      <td>0.019394</td>
      <td>-0.021064</td>
      <td>0.042260</td>
      <td>-0.102871</td>
      <td>-0.016305</td>
      <td>0.051671</td>
      <td>0.023082</td>
      <td>-0.009407</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>0.378195</td>
      <td>0.146402</td>
      <td>-0.006341</td>
      <td>0.082470</td>
      <td>0.291254</td>
      <td>0.058855</td>
      <td>0.165326</td>
      <td>0.013884</td>
      <td>-0.007214</td>
      <td>0.138629</td>
      <td>...</td>
      <td>0.031896</td>
      <td>0.038435</td>
      <td>-0.030948</td>
      <td>-0.009616</td>
      <td>0.000448</td>
      <td>0.024405</td>
      <td>-0.030815</td>
      <td>-0.007497</td>
      <td>-0.029901</td>
      <td>0.004695</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>0.386723</td>
      <td>0.436593</td>
      <td>0.029302</td>
      <td>0.333589</td>
      <td>0.067158</td>
      <td>-0.076072</td>
      <td>-0.027273</td>
      <td>-0.041978</td>
      <td>0.283160</td>
      <td>0.104650</td>
      <td>...</td>
      <td>0.014768</td>
      <td>-0.005304</td>
      <td>-0.008799</td>
      <td>0.001570</td>
      <td>-0.018359</td>
      <td>0.244249</td>
      <td>-0.019353</td>
      <td>-0.165823</td>
      <td>0.045597</td>
      <td>-0.026000</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>0.489782</td>
      <td>0.138347</td>
      <td>0.035087</td>
      <td>0.049242</td>
      <td>0.296970</td>
      <td>-0.018227</td>
      <td>0.140714</td>
      <td>-0.028726</td>
      <td>0.183303</td>
      <td>0.230741</td>
      <td>...</td>
      <td>-0.038068</td>
      <td>0.012220</td>
      <td>-0.002572</td>
      <td>-0.005535</td>
      <td>-0.041904</td>
      <td>0.296671</td>
      <td>-0.080601</td>
      <td>-0.218665</td>
      <td>-0.047794</td>
      <td>0.006372</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>0.456408</td>
      <td>0.190211</td>
      <td>0.019470</td>
      <td>0.102994</td>
      <td>0.220352</td>
      <td>-0.061106</td>
      <td>0.106398</td>
      <td>-0.022507</td>
      <td>0.250375</td>
      <td>0.170837</td>
      <td>...</td>
      <td>-0.039236</td>
      <td>0.004583</td>
      <td>-0.003825</td>
      <td>0.011800</td>
      <td>-0.038201</td>
      <td>0.327162</td>
      <td>-0.058890</td>
      <td>-0.231757</td>
      <td>-0.032854</td>
      <td>0.003240</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>0.166642</td>
      <td>0.064402</td>
      <td>0.029401</td>
      <td>-0.051575</td>
      <td>0.115843</td>
      <td>0.035070</td>
      <td>0.002365</td>
      <td>-0.004503</td>
      <td>0.042720</td>
      <td>0.265618</td>
      <td>...</td>
      <td>0.009940</td>
      <td>0.010801</td>
      <td>-0.056564</td>
      <td>-0.035304</td>
      <td>-0.038328</td>
      <td>0.070043</td>
      <td>-0.121200</td>
      <td>-0.036438</td>
      <td>0.032469</td>
      <td>0.005152</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>0.566024</td>
      <td>0.687501</td>
      <td>0.020643</td>
      <td>0.527567</td>
      <td>0.208171</td>
      <td>-0.009640</td>
      <td>0.044656</td>
      <td>-0.022963</td>
      <td>0.240257</td>
      <td>0.093666</td>
      <td>...</td>
      <td>0.030312</td>
      <td>0.008287</td>
      <td>-0.016628</td>
      <td>0.017268</td>
      <td>-0.022348</td>
      <td>0.168368</td>
      <td>-0.036522</td>
      <td>-0.121102</td>
      <td>0.044121</td>
      <td>-0.008545</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>-0.119916</td>
      <td>0.609707</td>
      <td>-0.004972</td>
      <td>0.210540</td>
      <td>0.004262</td>
      <td>-0.032148</td>
      <td>-0.007462</td>
      <td>0.001547</td>
      <td>-0.041118</td>
      <td>0.134637</td>
      <td>...</td>
      <td>-0.013854</td>
      <td>0.045466</td>
      <td>-0.059983</td>
      <td>0.001997</td>
      <td>-0.021325</td>
      <td>0.060505</td>
      <td>-0.034560</td>
      <td>-0.008467</td>
      <td>0.027628</td>
      <td>-0.019939</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>0.068101</td>
      <td>0.059306</td>
      <td>-0.024600</td>
      <td>0.265391</td>
      <td>-0.081007</td>
      <td>-0.040751</td>
      <td>0.025907</td>
      <td>-0.058649</td>
      <td>0.030086</td>
      <td>-0.246797</td>
      <td>...</td>
      <td>-0.011083</td>
      <td>-0.007832</td>
      <td>0.023075</td>
      <td>0.040833</td>
      <td>-0.012396</td>
      <td>-0.041377</td>
      <td>0.059075</td>
      <td>0.009080</td>
      <td>0.013583</td>
      <td>-0.005536</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>0.299475</td>
      <td>0.050986</td>
      <td>0.020423</td>
      <td>0.113681</td>
      <td>0.214103</td>
      <td>0.111170</td>
      <td>0.227786</td>
      <td>0.038502</td>
      <td>-0.002618</td>
      <td>0.049755</td>
      <td>...</td>
      <td>-0.007818</td>
      <td>-0.002872</td>
      <td>-0.006018</td>
      <td>0.001076</td>
      <td>-0.015040</td>
      <td>0.020039</td>
      <td>-0.005722</td>
      <td>-0.002292</td>
      <td>-0.197131</td>
      <td>0.010123</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>0.245181</td>
      <td>0.042549</td>
      <td>0.023499</td>
      <td>0.121093</td>
      <td>0.076670</td>
      <td>-0.009312</td>
      <td>0.029931</td>
      <td>-0.028578</td>
      <td>0.160829</td>
      <td>-0.011683</td>
      <td>...</td>
      <td>0.014939</td>
      <td>-0.006010</td>
      <td>0.031665</td>
      <td>0.005374</td>
      <td>-0.011881</td>
      <td>0.183706</td>
      <td>0.001366</td>
      <td>-0.139867</td>
      <td>-0.025107</td>
      <td>-0.043535</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>-0.014241</td>
      <td>0.063353</td>
      <td>-0.004296</td>
      <td>0.149313</td>
      <td>-0.064503</td>
      <td>0.014807</td>
      <td>-0.042304</td>
      <td>-0.006376</td>
      <td>0.028167</td>
      <td>-0.050149</td>
      <td>...</td>
      <td>-0.006302</td>
      <td>-0.004453</td>
      <td>0.082887</td>
      <td>-0.007049</td>
      <td>-0.007049</td>
      <td>-0.036308</td>
      <td>-0.005456</td>
      <td>0.025586</td>
      <td>0.007724</td>
      <td>-0.003148</td>
    </tr>
    <tr>
      <th>MSSubClass</th>
      <td>-0.251758</td>
      <td>0.307886</td>
      <td>-0.043825</td>
      <td>0.041622</td>
      <td>-0.069836</td>
      <td>-0.065649</td>
      <td>0.023578</td>
      <td>0.009469</td>
      <td>-0.140759</td>
      <td>-0.101774</td>
      <td>...</td>
      <td>0.028636</td>
      <td>0.028994</td>
      <td>0.085451</td>
      <td>-0.001244</td>
      <td>0.014005</td>
      <td>-0.045156</td>
      <td>-0.014555</td>
      <td>0.026359</td>
      <td>-0.024969</td>
      <td>-0.022844</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>0.339850</td>
      <td>0.173800</td>
      <td>0.019144</td>
      <td>0.093924</td>
      <td>0.261256</td>
      <td>-0.071330</td>
      <td>0.059590</td>
      <td>0.027440</td>
      <td>0.113862</td>
      <td>0.126409</td>
      <td>...</td>
      <td>-0.017731</td>
      <td>-0.021139</td>
      <td>0.001547</td>
      <td>0.015601</td>
      <td>-0.022686</td>
      <td>0.165692</td>
      <td>-0.025899</td>
      <td>-0.128187</td>
      <td>0.017108</td>
      <td>0.063452</td>
    </tr>
    <tr>
      <th>MiscVal</th>
      <td>-0.021096</td>
      <td>0.016197</td>
      <td>0.000354</td>
      <td>0.011634</td>
      <td>0.003571</td>
      <td>0.004940</td>
      <td>-0.016241</td>
      <td>-0.007392</td>
      <td>-0.023837</td>
      <td>-0.002478</td>
      <td>...</td>
      <td>-0.004596</td>
      <td>-0.003248</td>
      <td>0.002975</td>
      <td>0.013771</td>
      <td>-0.005140</td>
      <td>-0.026478</td>
      <td>-0.003979</td>
      <td>0.025009</td>
      <td>-0.022733</td>
      <td>-0.002296</td>
    </tr>
    <tr>
      <th>MoSold</th>
      <td>0.031372</td>
      <td>0.035164</td>
      <td>0.029474</td>
      <td>0.046933</td>
      <td>-0.015727</td>
      <td>-0.015211</td>
      <td>-0.028738</td>
      <td>0.027816</td>
      <td>0.034888</td>
      <td>0.009846</td>
      <td>...</td>
      <td>0.003454</td>
      <td>-0.011263</td>
      <td>0.016522</td>
      <td>0.040735</td>
      <td>-0.054700</td>
      <td>0.094991</td>
      <td>0.028174</td>
      <td>-0.087446</td>
      <td>0.003690</td>
      <td>-0.051552</td>
    </tr>
    <tr>
      <th>OpenPorchSF</th>
      <td>0.211671</td>
      <td>0.208026</td>
      <td>-0.005842</td>
      <td>0.090935</td>
      <td>0.111761</td>
      <td>0.003093</td>
      <td>0.055175</td>
      <td>-0.028273</td>
      <td>0.129005</td>
      <td>0.025858</td>
      <td>...</td>
      <td>-0.021098</td>
      <td>-0.005122</td>
      <td>-0.030380</td>
      <td>0.023489</td>
      <td>-0.019525</td>
      <td>0.171467</td>
      <td>-0.025573</td>
      <td>-0.106605</td>
      <td>-0.005664</td>
      <td>0.028199</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>-0.144203</td>
      <td>0.028942</td>
      <td>0.025504</td>
      <td>0.003265</td>
      <td>-0.046231</td>
      <td>0.040229</td>
      <td>-0.054147</td>
      <td>0.098780</td>
      <td>-0.136841</td>
      <td>0.118969</td>
      <td>...</td>
      <td>0.031788</td>
      <td>-0.019156</td>
      <td>-0.064332</td>
      <td>0.001299</td>
      <td>-0.019779</td>
      <td>-0.156175</td>
      <td>-0.050663</td>
      <td>0.163684</td>
      <td>0.042848</td>
      <td>0.009994</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>0.476224</td>
      <td>0.295493</td>
      <td>0.030371</td>
      <td>0.082689</td>
      <td>0.239666</td>
      <td>-0.059119</td>
      <td>0.084653</td>
      <td>-0.032511</td>
      <td>0.308159</td>
      <td>0.272038</td>
      <td>...</td>
      <td>0.034147</td>
      <td>0.037524</td>
      <td>-0.037305</td>
      <td>0.004269</td>
      <td>-0.021172</td>
      <td>0.327412</td>
      <td>-0.057962</td>
      <td>-0.225013</td>
      <td>0.058823</td>
      <td>-0.001881</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>0.131525</td>
      <td>0.081487</td>
      <td>-0.007992</td>
      <td>0.072988</td>
      <td>0.140491</td>
      <td>0.041709</td>
      <td>0.076760</td>
      <td>0.016983</td>
      <td>-0.035092</td>
      <td>0.018122</td>
      <td>...</td>
      <td>-0.003600</td>
      <td>-0.002544</td>
      <td>-0.005410</td>
      <td>-0.004027</td>
      <td>-0.004027</td>
      <td>0.008838</td>
      <td>-0.003117</td>
      <td>0.002642</td>
      <td>0.004413</td>
      <td>-0.001798</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>PavedDrive_P</th>
      <td>-0.062613</td>
      <td>0.012140</td>
      <td>-0.016851</td>
      <td>0.010184</td>
      <td>-0.079791</td>
      <td>-0.041809</td>
      <td>-0.055867</td>
      <td>0.021285</td>
      <td>0.024459</td>
      <td>-0.079237</td>
      <td>...</td>
      <td>-0.007592</td>
      <td>-0.005364</td>
      <td>-0.011407</td>
      <td>-0.008491</td>
      <td>-0.008491</td>
      <td>-0.043737</td>
      <td>-0.006572</td>
      <td>0.056531</td>
      <td>0.009304</td>
      <td>-0.003792</td>
    </tr>
    <tr>
      <th>PavedDrive_Y</th>
      <td>0.163848</td>
      <td>-0.040082</td>
      <td>0.022902</td>
      <td>-0.054431</td>
      <td>0.191902</td>
      <td>0.068581</td>
      <td>0.092340</td>
      <td>0.029823</td>
      <td>-0.015493</td>
      <td>0.325482</td>
      <td>...</td>
      <td>0.015685</td>
      <td>0.011083</td>
      <td>-0.072014</td>
      <td>0.017543</td>
      <td>-0.025144</td>
      <td>0.081351</td>
      <td>-0.041492</td>
      <td>-0.065257</td>
      <td>0.019757</td>
      <td>0.007834</td>
    </tr>
    <tr>
      <th>PoolQC_Yes</th>
      <td>0.146727</td>
      <td>0.090073</td>
      <td>-0.008075</td>
      <td>0.074962</td>
      <td>0.166271</td>
      <td>0.043296</td>
      <td>0.089494</td>
      <td>0.021204</td>
      <td>-0.037279</td>
      <td>0.018311</td>
      <td>...</td>
      <td>-0.003638</td>
      <td>-0.002571</td>
      <td>-0.005466</td>
      <td>-0.004069</td>
      <td>-0.004069</td>
      <td>0.014872</td>
      <td>-0.003150</td>
      <td>-0.002186</td>
      <td>0.004459</td>
      <td>-0.001817</td>
    </tr>
    <tr>
      <th>RoofMatl_CompShg</th>
      <td>-0.190345</td>
      <td>-0.010648</td>
      <td>-0.007307</td>
      <td>-0.006294</td>
      <td>-0.105754</td>
      <td>-0.093091</td>
      <td>-0.114261</td>
      <td>-0.104081</td>
      <td>0.022127</td>
      <td>-0.014526</td>
      <td>...</td>
      <td>0.007058</td>
      <td>0.004987</td>
      <td>0.010605</td>
      <td>0.007893</td>
      <td>0.007893</td>
      <td>0.021945</td>
      <td>0.006110</td>
      <td>-0.021972</td>
      <td>-0.008650</td>
      <td>0.003525</td>
    </tr>
    <tr>
      <th>RoofMatl_Membran</th>
      <td>0.013574</td>
      <td>-0.020818</td>
      <td>-0.003046</td>
      <td>-0.025540</td>
      <td>-0.012497</td>
      <td>0.165014</td>
      <td>0.022216</td>
      <td>-0.005837</td>
      <td>-0.027930</td>
      <td>0.006907</td>
      <td>...</td>
      <td>-0.001372</td>
      <td>-0.000970</td>
      <td>-0.002062</td>
      <td>-0.001535</td>
      <td>-0.001535</td>
      <td>-0.007905</td>
      <td>-0.001188</td>
      <td>0.010218</td>
      <td>0.001682</td>
      <td>-0.000685</td>
    </tr>
    <tr>
      <th>RoofMatl_Metal</th>
      <td>-0.011830</td>
      <td>-0.020818</td>
      <td>0.113083</td>
      <td>-0.041236</td>
      <td>0.028386</td>
      <td>-0.007557</td>
      <td>0.022216</td>
      <td>-0.005837</td>
      <td>-0.033620</td>
      <td>0.006907</td>
      <td>...</td>
      <td>-0.001372</td>
      <td>-0.000970</td>
      <td>-0.002062</td>
      <td>-0.001535</td>
      <td>-0.001535</td>
      <td>-0.007905</td>
      <td>-0.001188</td>
      <td>0.010218</td>
      <td>0.001682</td>
      <td>-0.000685</td>
    </tr>
    <tr>
      <th>RoofMatl_Roll</th>
      <td>-0.015895</td>
      <td>0.038697</td>
      <td>-0.003046</td>
      <td>0.037244</td>
      <td>-0.012841</td>
      <td>-0.007557</td>
      <td>-0.018176</td>
      <td>-0.005837</td>
      <td>0.008343</td>
      <td>0.006907</td>
      <td>...</td>
      <td>-0.001372</td>
      <td>-0.000970</td>
      <td>-0.002062</td>
      <td>-0.001535</td>
      <td>-0.001535</td>
      <td>-0.007905</td>
      <td>-0.001188</td>
      <td>0.010218</td>
      <td>0.001682</td>
      <td>-0.000685</td>
    </tr>
    <tr>
      <th>RoofMatl_Tar&amp;Grv</th>
      <td>0.071021</td>
      <td>-0.023777</td>
      <td>-0.010137</td>
      <td>0.009978</td>
      <td>0.015044</td>
      <td>0.088310</td>
      <td>0.037273</td>
      <td>0.156375</td>
      <td>-0.047352</td>
      <td>-0.009128</td>
      <td>...</td>
      <td>-0.004567</td>
      <td>-0.003227</td>
      <td>-0.006862</td>
      <td>-0.005108</td>
      <td>-0.005108</td>
      <td>-0.026310</td>
      <td>-0.003954</td>
      <td>0.034006</td>
      <td>0.005597</td>
      <td>-0.002281</td>
    </tr>
    <tr>
      <th>RoofMatl_WdShake</th>
      <td>0.096561</td>
      <td>0.009270</td>
      <td>-0.006820</td>
      <td>-0.005641</td>
      <td>0.005266</td>
      <td>0.012160</td>
      <td>0.049744</td>
      <td>-0.013070</td>
      <td>0.011726</td>
      <td>0.015465</td>
      <td>...</td>
      <td>-0.003073</td>
      <td>-0.002171</td>
      <td>-0.004617</td>
      <td>-0.003436</td>
      <td>-0.003436</td>
      <td>-0.017701</td>
      <td>-0.002660</td>
      <td>-0.011736</td>
      <td>0.003766</td>
      <td>-0.001535</td>
    </tr>
    <tr>
      <th>RoofMatl_WdShngl</th>
      <td>0.117333</td>
      <td>0.032092</td>
      <td>-0.007473</td>
      <td>0.016499</td>
      <td>0.070121</td>
      <td>0.003765</td>
      <td>0.071029</td>
      <td>0.025282</td>
      <td>0.031765</td>
      <td>0.016947</td>
      <td>...</td>
      <td>-0.003367</td>
      <td>-0.002379</td>
      <td>-0.005059</td>
      <td>-0.003766</td>
      <td>-0.003766</td>
      <td>-0.019397</td>
      <td>-0.002915</td>
      <td>0.025072</td>
      <td>0.004127</td>
      <td>-0.001682</td>
    </tr>
    <tr>
      <th>RoofStyle_Gable</th>
      <td>-0.314131</td>
      <td>0.086670</td>
      <td>-0.029938</td>
      <td>-0.013846</td>
      <td>-0.193130</td>
      <td>-0.070168</td>
      <td>-0.080428</td>
      <td>-0.029263</td>
      <td>-0.027912</td>
      <td>-0.005086</td>
      <td>...</td>
      <td>-0.003996</td>
      <td>0.019583</td>
      <td>-0.000711</td>
      <td>0.030996</td>
      <td>0.030996</td>
      <td>-0.055967</td>
      <td>0.023993</td>
      <td>0.038323</td>
      <td>0.017853</td>
      <td>0.013843</td>
    </tr>
    <tr>
      <th>RoofStyle_Gambrel</th>
      <td>-0.061752</td>
      <td>0.071027</td>
      <td>-0.010137</td>
      <td>0.038471</td>
      <td>-0.053247</td>
      <td>-0.017782</td>
      <td>-0.036051</td>
      <td>0.009874</td>
      <td>0.004991</td>
      <td>-0.073356</td>
      <td>...</td>
      <td>-0.004567</td>
      <td>-0.003227</td>
      <td>-0.006862</td>
      <td>-0.005108</td>
      <td>-0.005108</td>
      <td>-0.026310</td>
      <td>-0.003954</td>
      <td>0.034006</td>
      <td>0.005597</td>
      <td>-0.002281</td>
    </tr>
    <tr>
      <th>RoofStyle_Hip</th>
      <td>0.323994</td>
      <td>-0.113568</td>
      <td>0.030141</td>
      <td>0.000307</td>
      <td>0.213285</td>
      <td>0.033034</td>
      <td>0.078014</td>
      <td>-0.001522</td>
      <td>0.044537</td>
      <td>0.025256</td>
      <td>...</td>
      <td>0.007146</td>
      <td>-0.018280</td>
      <td>0.005225</td>
      <td>-0.028934</td>
      <td>-0.028934</td>
      <td>0.075468</td>
      <td>-0.022396</td>
      <td>-0.062128</td>
      <td>-0.022246</td>
      <td>-0.012922</td>
    </tr>
    <tr>
      <th>RoofStyle_Mansard</th>
      <td>0.000529</td>
      <td>0.073463</td>
      <td>-0.008075</td>
      <td>0.065054</td>
      <td>-0.048464</td>
      <td>0.008372</td>
      <td>-0.048189</td>
      <td>0.021204</td>
      <td>0.010849</td>
      <td>-0.021891</td>
      <td>...</td>
      <td>-0.003638</td>
      <td>-0.002571</td>
      <td>-0.005466</td>
      <td>-0.004069</td>
      <td>-0.004069</td>
      <td>-0.020959</td>
      <td>-0.003150</td>
      <td>0.027090</td>
      <td>0.004459</td>
      <td>-0.001817</td>
    </tr>
    <tr>
      <th>RoofStyle_Shed</th>
      <td>0.017622</td>
      <td>0.032125</td>
      <td>-0.004309</td>
      <td>-0.006525</td>
      <td>0.035284</td>
      <td>0.013539</td>
      <td>0.088571</td>
      <td>-0.008258</td>
      <td>-0.017125</td>
      <td>0.009771</td>
      <td>...</td>
      <td>-0.001941</td>
      <td>-0.001372</td>
      <td>-0.002917</td>
      <td>-0.002171</td>
      <td>-0.002171</td>
      <td>-0.011184</td>
      <td>-0.001681</td>
      <td>0.014455</td>
      <td>0.002379</td>
      <td>-0.000970</td>
    </tr>
    <tr>
      <th>SaleCondition_AdjLand</th>
      <td>-0.037451</td>
      <td>-0.014533</td>
      <td>-0.006098</td>
      <td>0.048378</td>
      <td>-0.014874</td>
      <td>-0.015130</td>
      <td>-0.016173</td>
      <td>0.182202</td>
      <td>-0.034618</td>
      <td>-0.092426</td>
      <td>...</td>
      <td>-0.002747</td>
      <td>-0.001941</td>
      <td>-0.004128</td>
      <td>-0.003073</td>
      <td>-0.003073</td>
      <td>-0.015827</td>
      <td>-0.002378</td>
      <td>0.020457</td>
      <td>0.003367</td>
      <td>-0.001372</td>
    </tr>
    <tr>
      <th>SaleCondition_Alloca</th>
      <td>0.068107</td>
      <td>-0.020234</td>
      <td>-0.010591</td>
      <td>0.052186</td>
      <td>0.021369</td>
      <td>-0.026277</td>
      <td>0.194292</td>
      <td>-0.020297</td>
      <td>-0.059130</td>
      <td>-0.006741</td>
      <td>...</td>
      <td>-0.004772</td>
      <td>-0.003372</td>
      <td>-0.007170</td>
      <td>-0.005337</td>
      <td>-0.005337</td>
      <td>-0.027489</td>
      <td>-0.004131</td>
      <td>0.035530</td>
      <td>-0.112734</td>
      <td>-0.002383</td>
    </tr>
    <tr>
      <th>SaleCondition_Family</th>
      <td>0.021949</td>
      <td>-0.027180</td>
      <td>-0.013711</td>
      <td>0.066381</td>
      <td>0.000765</td>
      <td>-0.007929</td>
      <td>-0.018183</td>
      <td>0.039116</td>
      <td>0.021534</td>
      <td>-0.016691</td>
      <td>...</td>
      <td>0.106555</td>
      <td>-0.004365</td>
      <td>-0.009282</td>
      <td>-0.006909</td>
      <td>-0.006909</td>
      <td>-0.035587</td>
      <td>-0.005348</td>
      <td>0.028599</td>
      <td>0.007571</td>
      <td>-0.003085</td>
    </tr>
    <tr>
      <th>SaleCondition_Normal</th>
      <td>-0.158772</td>
      <td>0.031766</td>
      <td>-0.009177</td>
      <td>0.003551</td>
      <td>-0.019560</td>
      <td>0.041207</td>
      <td>-0.022306</td>
      <td>-0.034388</td>
      <td>-0.153930</td>
      <td>-0.014821</td>
      <td>...</td>
      <td>-0.043784</td>
      <td>0.017320</td>
      <td>-0.031583</td>
      <td>-0.003139</td>
      <td>0.027414</td>
      <td>-0.645698</td>
      <td>-0.097031</td>
      <td>0.634322</td>
      <td>-0.002140</td>
      <td>-0.055982</td>
    </tr>
    <tr>
      <th>SaleCondition_Partial</th>
      <td>0.221037</td>
      <td>0.004852</td>
      <td>0.018526</td>
      <td>-0.060265</td>
      <td>0.044912</td>
      <td>-0.085761</td>
      <td>-0.004721</td>
      <td>-0.022949</td>
      <td>0.249315</td>
      <td>0.080725</td>
      <td>...</td>
      <td>-0.016038</td>
      <td>-0.011333</td>
      <td>0.007176</td>
      <td>-0.017938</td>
      <td>-0.017938</td>
      <td>0.986819</td>
      <td>-0.013885</td>
      <td>-0.769559</td>
      <td>0.019657</td>
      <td>-0.008011</td>
    </tr>
    <tr>
      <th>SaleType_CWD</th>
      <td>0.033381</td>
      <td>0.007628</td>
      <td>-0.006098</td>
      <td>0.030047</td>
      <td>0.008951</td>
      <td>0.076364</td>
      <td>-0.016173</td>
      <td>0.036786</td>
      <td>-0.028685</td>
      <td>-0.039299</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.001941</td>
      <td>-0.004128</td>
      <td>-0.003073</td>
      <td>-0.003073</td>
      <td>-0.015827</td>
      <td>-0.002378</td>
      <td>-0.134295</td>
      <td>0.003367</td>
      <td>-0.001372</td>
    </tr>
    <tr>
      <th>SaleType_Con</th>
      <td>0.007559</td>
      <td>0.003778</td>
      <td>-0.004309</td>
      <td>-0.028730</td>
      <td>0.030694</td>
      <td>-0.010691</td>
      <td>0.088571</td>
      <td>-0.008258</td>
      <td>-0.012681</td>
      <td>0.009771</td>
      <td>...</td>
      <td>-0.001941</td>
      <td>1.000000</td>
      <td>-0.002917</td>
      <td>-0.002171</td>
      <td>-0.002171</td>
      <td>-0.011184</td>
      <td>-0.001681</td>
      <td>-0.094896</td>
      <td>0.002379</td>
      <td>-0.000970</td>
    </tr>
    <tr>
      <th>SaleType_ConLD</th>
      <td>-0.011789</td>
      <td>-0.018808</td>
      <td>-0.009162</td>
      <td>-0.013874</td>
      <td>-0.021376</td>
      <td>-0.022733</td>
      <td>-0.014176</td>
      <td>-0.017560</td>
      <td>-0.000835</td>
      <td>-0.085660</td>
      <td>...</td>
      <td>-0.004128</td>
      <td>-0.002917</td>
      <td>1.000000</td>
      <td>-0.004617</td>
      <td>-0.004617</td>
      <td>-0.023782</td>
      <td>-0.003574</td>
      <td>-0.201789</td>
      <td>-0.131726</td>
      <td>-0.002062</td>
    </tr>
    <tr>
      <th>SaleType_ConLI</th>
      <td>0.006094</td>
      <td>0.016175</td>
      <td>-0.006820</td>
      <td>0.081051</td>
      <td>0.022726</td>
      <td>-0.016921</td>
      <td>-0.004522</td>
      <td>-0.013070</td>
      <td>0.001853</td>
      <td>0.015465</td>
      <td>...</td>
      <td>-0.003073</td>
      <td>-0.002171</td>
      <td>-0.004617</td>
      <td>1.000000</td>
      <td>-0.003436</td>
      <td>-0.017701</td>
      <td>-0.002660</td>
      <td>-0.150198</td>
      <td>0.003766</td>
      <td>-0.001535</td>
    </tr>
    <tr>
      <th>SaleType_ConLw</th>
      <td>-0.043721</td>
      <td>0.012602</td>
      <td>-0.006820</td>
      <td>-0.029071</td>
      <td>-0.017825</td>
      <td>0.035715</td>
      <td>-0.004522</td>
      <td>0.030299</td>
      <td>-0.033900</td>
      <td>-0.079604</td>
      <td>...</td>
      <td>-0.003073</td>
      <td>-0.002171</td>
      <td>-0.004617</td>
      <td>-0.003436</td>
      <td>1.000000</td>
      <td>-0.017701</td>
      <td>-0.002660</td>
      <td>-0.150198</td>
      <td>0.003766</td>
      <td>-0.001535</td>
    </tr>
    <tr>
      <th>SaleType_New</th>
      <td>0.221219</td>
      <td>0.010810</td>
      <td>0.019596</td>
      <td>-0.054679</td>
      <td>0.044883</td>
      <td>-0.087162</td>
      <td>-0.014893</td>
      <td>-0.021547</td>
      <td>0.249236</td>
      <td>0.079661</td>
      <td>...</td>
      <td>-0.015827</td>
      <td>-0.011184</td>
      <td>-0.023782</td>
      <td>-0.017701</td>
      <td>-0.017701</td>
      <td>1.000000</td>
      <td>-0.013702</td>
      <td>-0.773680</td>
      <td>0.019397</td>
      <td>-0.007905</td>
    </tr>
    <tr>
      <th>SaleType_Oth</th>
      <td>-0.008215</td>
      <td>-0.036082</td>
      <td>-0.005279</td>
      <td>0.022234</td>
      <td>0.010652</td>
      <td>-0.013098</td>
      <td>0.061841</td>
      <td>-0.010117</td>
      <td>-0.002593</td>
      <td>0.011971</td>
      <td>...</td>
      <td>-0.002378</td>
      <td>-0.001681</td>
      <td>-0.003574</td>
      <td>-0.002660</td>
      <td>-0.002660</td>
      <td>-0.013702</td>
      <td>1.000000</td>
      <td>-0.116263</td>
      <td>0.002915</td>
      <td>-0.001188</td>
    </tr>
    <tr>
      <th>SaleType_WD</th>
      <td>-0.198056</td>
      <td>0.026769</td>
      <td>-0.014211</td>
      <td>0.046933</td>
      <td>-0.024778</td>
      <td>0.036178</td>
      <td>0.015132</td>
      <td>0.012215</td>
      <td>-0.198960</td>
      <td>-0.037373</td>
      <td>...</td>
      <td>-0.134295</td>
      <td>-0.094896</td>
      <td>-0.201789</td>
      <td>-0.150198</td>
      <td>-0.150198</td>
      <td>-0.773680</td>
      <td>-0.116263</td>
      <td>1.000000</td>
      <td>0.006539</td>
      <td>-0.067078</td>
    </tr>
    <tr>
      <th>Street_Pave</th>
      <td>0.005950</td>
      <td>0.046983</td>
      <td>0.007473</td>
      <td>0.030573</td>
      <td>-0.015643</td>
      <td>-0.038487</td>
      <td>-0.071029</td>
      <td>0.014323</td>
      <td>0.035229</td>
      <td>0.069869</td>
      <td>...</td>
      <td>0.003367</td>
      <td>0.002379</td>
      <td>-0.131726</td>
      <td>0.003766</td>
      <td>0.003766</td>
      <td>0.019397</td>
      <td>0.002915</td>
      <td>0.006539</td>
      <td>1.000000</td>
      <td>0.001682</td>
    </tr>
    <tr>
      <th>Utilities_NoSeWa</th>
      <td>0.012287</td>
      <td>-0.020818</td>
      <td>-0.003046</td>
      <td>0.000620</td>
      <td>-0.019100</td>
      <td>0.049913</td>
      <td>-0.018176</td>
      <td>0.091007</td>
      <td>-0.012639</td>
      <td>0.006907</td>
      <td>...</td>
      <td>-0.001372</td>
      <td>-0.000970</td>
      <td>-0.002062</td>
      <td>-0.001535</td>
      <td>-0.001535</td>
      <td>-0.007905</td>
      <td>-0.001188</td>
      <td>-0.067078</td>
      <td>0.001682</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>242 rows × 242 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt"></div>

<div class="output_subarea output_stream output_stderr output_text">
<pre class="editor-colors lang-text"></pre>
</div>
</div>

<div class="output_area">
<div class="prompt"></div>

<div class="output_subarea output_stream output_stdout output_text">
<pre class="editor-colors lang-text"></pre>
</div>
</div>

<div class="output_area">
<div class="prompt output_prompt">Out[3]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt"></div>

<div class="output_subarea output_stream output_stderr output_text">
<pre class="editor-colors lang-text"></pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt"></div>

<div class="output_subarea output_stream output_stdout output_text">
<pre class="editor-colors lang-text"></pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt"></div>

<div class="output_subarea output_stream output_stderr output_text">
<pre class="editor-colors lang-text"></pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[7]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[8]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt"></div>

<div class="output_subarea output_stream output_stderr output_text">
<pre class="editor-colors lang-text"><div class="line"><span class="syntax--text syntax--plain"><span class="syntax--meta syntax--paragraph syntax--text"><span>main</span></span></span></div></pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[10]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[11]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[12]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt"></div>

<div class="output_subarea output_stream output_stdout output_text">
<pre class="editor-colors lang-text"></pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[14]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[15]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt"></div>

<div class="output_subarea output_stream output_stderr output_text">
<pre class="editor-colors lang-text"></pre>
</div>
</div>

<div class="output_area">
<div class="prompt output_prompt">Out[16]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[17]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[18]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[19]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[21]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[22]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[23]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt"></div>

<div class="output_subarea output_stream output_stdout output_text">
<pre class="editor-colors lang-text"></pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[28]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[29]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[29]:</div>


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>3SsnPorch</th>
      <th>BedroomAbvGr</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>BsmtUnfSF</th>
      <th>CentralAir</th>
      <th>...</th>
      <th>SaleType_CWD</th>
      <th>SaleType_Con</th>
      <th>SaleType_ConLD</th>
      <th>SaleType_ConLI</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>Street_Pave</th>
      <th>Utilities_NoSeWa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>896</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>468.0</td>
      <td>144.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>270.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1329</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>923.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>406.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>928</td>
      <td>701</td>
      <td>0</td>
      <td>9</td>
      <td>791.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>926</td>
      <td>678</td>
      <td>0</td>
      <td>9</td>
      <td>602.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>324.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1280</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>263.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1017.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 241 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[30]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[30]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[31]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[31]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[32]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[33]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[33]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[34]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[34]:</div>



<div class="output_text output_subarea output_execute_result">
<pre class="editor-colors lang-text"></pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[37]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt"></div>

<div class="output_subarea output_stream output_stderr output_text">
<pre class="editor-colors lang-text"><div class="line"><span class="syntax--text syntax--plain"><span class="syntax--meta syntax--paragraph syntax--text"><span>main</span></span></span></div></pre>
</div>
</div>

<div class="output_area">
<div class="prompt output_prompt">Out[37]:</div>


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1439</th>
      <td>2900</td>
      <td>155000.0</td>
    </tr>
    <tr>
      <th>1440</th>
      <td>2901</td>
      <td>168000.0</td>
    </tr>
    <tr>
      <th>1441</th>
      <td>2902</td>
      <td>150000.0</td>
    </tr>
    <tr>
      <th>1442</th>
      <td>2903</td>
      <td>190000.0</td>
    </tr>
    <tr>
      <th>1443</th>
      <td>2904</td>
      <td>325624.0</td>
    </tr>
    <tr>
      <th>1444</th>
      <td>2905</td>
      <td>131400.0</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>2906</td>
      <td>155000.0</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>2907</td>
      <td>107500.0</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>2908</td>
      <td>147000.0</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>2909</td>
      <td>146800.0</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>2910</td>
      <td>110500.0</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>2911</td>
      <td>106000.0</td>
    </tr>
    <tr>
      <th>1451</th>
      <td>2912</td>
      <td>139000.0</td>
    </tr>
    <tr>
      <th>1452</th>
      <td>2913</td>
      <td>106000.0</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>2914</td>
      <td>97000.0</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>2915</td>
      <td>97000.0</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>2916</td>
      <td>106000.0</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>2917</td>
      <td>149900.0</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>2918</td>
      <td>190000.0</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>2919</td>
      <td>185000.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[36]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre class="editor-colors lang-text"></pre></div>

</div>
</div>
</div>

</div>
    </div>
  </div>
</body>




</html>
