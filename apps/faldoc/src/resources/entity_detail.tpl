<h1 class="faldoc_title"><%= title%><%if module_name%><span class="faldoc_belong"><a href="<%= module_link%>"><%= module_name%></a></p><%end%></h1>

<p class="faldoc_brief"><%= brief%></p>
<%if grammar%><p class="faldoc_funcdecl">
<%= grammar%>
</p><%end%>

<%if sections%>
   <%if description%><p class="faldoc_brief"><a href="#more">more...</a></p><%end%>
   <h2 class="faldoc_title">Summary</h2>
   <table class="faldoc_list">
   <%for sections%>
      <%for items%>
         <tr><td><a href="#<%= item_name%>"><%= item_title%></a></td><td><%= item_brief%></td></tr>
      <%end%>
   <%end%>
   </table>
<%end%>

<%if inherit_props%>
   <h3 class="faldoc_title">Inherited properties</h3>
   <table class="faldoc_list">
   <%for inherit_props%>
      <tr><td><a href="<%= prop_link%>"><%= prop_name%></a> from <%= prop_origin%>&nbsp;</td><td><%= prop_brief%></td></tr>
   <%end%>
   </table>
<%end%>

<%if inherit_methods%>
   <h3 class="faldoc_title">Inherited methods</h3>
   <table class="faldoc_list">
   <%for inherit_methods%>
      <tr><td><a href="<%= method_link%>"><%= method_name%></a> from <%= method_origin%>&nbsp;</td><td><%= method_brief%></td></tr>
   <%end%>
   </table>
<%end%>

<%if description%>
<a name="more"><h2 class="faldoc_title">Detailed description</h2></a>

<%if grammar%><p class="faldoc_funcdecl">
<%= grammar%>
</p><%end%>
<table class="faldoc_function">
<%for item_params %><tr><td class="faldoc_param"><%= param_name%></td><td class="faldoc_paramdesc"><%= param_desc%></td></tr><%end%>
<%for item_optparams %><tr><td class="faldoc_optparam"><%= param_name%></td><td class="faldoc_optparamdesc"><%= param_desc%></td></tr><%end%>
</table>
<br/>
<p class="item_brief"><%= brief%></p>
<%= description%>
<%end%>

<%for sections%>
   <h2 class="faldoc_title"><%= section_title%></h2>
   <%for items%>
      <h3 class="faldoc_funcname"><a name="<%= item_name%>"><%= item_title%></a></h3>
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
<%end%>
