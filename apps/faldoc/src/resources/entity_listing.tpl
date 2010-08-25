<h1 class="faldoc_title"><%if entity_type%><%= entity_type%> <%end%><%= entity_name%></h1>
<%if brief%><p class="faldoc_brief"><%= brief%></p><%end%>
<%if description%>
   <p class="faldoc_text"><%= description%></p>
   <%if items%>
      <h2 class="faldoc_title">Contents</h2>
   <%end%>
<%end%>

<%if items%>
   <table class="faldoc_list">
   <%for items%>
   <tr><td class="faldoc_funcname"><a href="<%= item_link%>"><%= item_name%></a></td><td class="faldoc_funcbrief"><%= item_brief%></td></tr>
   <%end%>
   </table>
<%end%>

<%for sections%>
   <h2 class="faldoc_title"><%= section_title%></h2>
   <%if section_description%><p class="faldoc_text"><%= section_description%></p><%end%>
   <table class="faldoc_list">
   <%for items%>
   <tr><td class="faldoc_funcname"><a href="<%= item_link%>"><%= item_name%></a></td><td class="faldoc_funcbrief"><%= item_brief%></td></tr>
   <%end%>
   </table>
<%end%>
