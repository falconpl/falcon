
<h1 class="faldoc_title"><%= page_title%></h1>
<%if description%><%= description%><%end%>

<%for items%>
   <h3 class="faldoc_funcname"><a name="<%= item_name%>"><%= item_title%></a><%if module_name%><span class="faldoc_belong"><a href="<%= module_link%>"><%= module_name%></a></p><%end%></h3>
   <p class="item_brief"><%= item_brief%></p>
   <%if item_grammar%><p class="faldoc_funcdecl"><%= item_grammar%></p><%end%>
   <%if item_descriptor_table%>
      <table class="faldoc_function">
      <%for item_params %><tr><td class="faldoc_param"><%= param_name%></td><td class="faldoc_paramdesc"><%= param_desc%></td></tr><%end%>
      <%for item_optparams %><tr><td class="faldoc_optparam"><%= param_name%></td><td class="faldoc_optparamdesc"><%= param_desc%></td></tr><%end%>
      <%if item_return %><tr><td class="faldoc_funcreturn">Returns:</td><td class="faldoc_funcreturndesc"><%= item_return %></td></tr><%end%>
      <%if item_raises %>
         <tr><td class="faldoc_raise">Raises:</td><td class="faldoc_raisedesc">
         <table class="faldoc_raise">
         <%for item_raises %><tr><td class="faldoc_raiseitem"><%= raise_name%></td><td class="faldoc_raisedesc"><%= raise_desc%></td></tr><%end%>
         </table>
         </td></tr>
      <%end%>
      </table>
   <%end%>
   <%if item_description%><p class="faldoc_text"><%= item_description%></p><%end%>
<%end%>
