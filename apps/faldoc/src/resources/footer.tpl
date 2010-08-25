<hr/>
<div class="navibottom">
   <center>
      <%for doclinks%><%forsep%>&nbsp;&nbsp;-&nbsp;&nbsp;<%end%><%if current%><b><%= l_desc%></b><%else%><a href="<%= l_ref%>"><%= l_desc%></a><%end%><%end%>
   </center>
</div>
</div>
<div class="faldoc_signature">Made with <a href="http://www.falconpl.org">faldoc <%= version%></div>
</body>
</html>
