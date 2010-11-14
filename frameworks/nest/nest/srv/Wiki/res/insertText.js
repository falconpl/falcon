function nest_wiki_insertText( text, mode )
{
   if( mode == 'image' )
      text = '{{'+text+'|An image|w=100|h=100|center|thumb|pad=20}}';
   else
      text = '[[:'+text+'|A file]]';

   obj = document.getElementById( "nest_wiki_area" );

   /* IE */
   if (document.all) {
      obj.focus();
      var sel=document.selection;
      var rng=sel.createRange();
      rng.colapse;
      rng.text= text;
   }
   /* Mozilla/geko */
   else if(obj.selectionEnd)
   {
      var lng=obj.textLength;
      var from=obj.selectionStart;
      var to=obj.selectionEnd;
      obj.value=obj.value.substring(0,from)+ text +obj.value.substring(to,lng);
   }
   /* No idea */
   else
      obj.value+=text;

   obj.focus();
}
