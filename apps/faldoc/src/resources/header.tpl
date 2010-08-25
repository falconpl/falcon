<html>
<head>
   <meta http-equiv="Content-Type" content="text/html<%if charset%>; charset=<%= charset%><%end%>" />
   <title><%= title%><%if heading%> - <%= heading%><%end%></title>
   <link href="faldoc.css" rel="stylesheet" type="text/css"/>
   <link href="tabs.css" rel="stylesheet" type="text/css"/>
</head>
<body class="faldoc_body">
<div class="navitop">
   <div class="tabs">
      <ul>
         <%for doclinks%><li<%if current%> class="current"<%end%>><a href="<%= l_ref%>"><span><%= l_desc%></span></a></li>
         <%end%>
      </ul>
   </div>
</div>
<hr/>

