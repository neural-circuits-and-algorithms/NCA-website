---
layout: page
permalink: /publications/
title: Publications
description: Publications out of the Chklovskii Lab
years: [2018, 2019, 2020, 2021, 2022, 2023]
nav: true
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years reversed %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
