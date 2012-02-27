/* Falcon - Nest framework - Special effexcts */

var Nest;
if(!Nest) { Nest = {}; }

(function () {
   "use strict";
   //========================================================== Private
   var transArray = new Array();
   var tempArray = new Array();
   var nextArray = new Array();
   var fps = 50;

   function getStyle(oElm, strCssRule){
      var strValue = "";
      if(document.defaultView && document.defaultView.getComputedStyle){
         strValue = document.defaultView.getComputedStyle(oElm, "").getPropertyValue(strCssRule);
      }
      else if(oElm.currentStyle){
         strCssRule = strCssRule.replace(/\-(\w)/g, function (strMatch, p1){
            return p1.toUpperCase();
         });
         strValue = oElm.currentStyle[strCssRule];
      }
      return strValue;
   }

   function getNumValue( string )
   {
      if( typeof string == 'string' ) {
         var value = string.match(/\|([0-9.]+)\|/);
         if( value )
            return parseInt(value[1]);
         return null;
      }

      return string;
   }

   function getNumFromValue( string, source )
   {
      if( typeof string == 'string' ) {
         var pos = string.search(/\|([0-9.]+)\|/);
         if( pos >= 0 )
            return parseInt(source.substr(pos));

      }

      return parseInt(source);
   }

   function setNumValue( string, value )
   {
      if( typeof string == 'string' ) {
         return string.replace(/\|([0-9.]+)\|/, value);
      }
      return value;
   }

   function getOriginalValue( string, value )
   {
      if( typeof string == 'string' ) {
         return string.replace(/\|/gi, "");
      }
      return value;
   }

   function progress() {
      // clear the temporary array
      if( tempArray.length > 0 ) tempArray.splice( 0, tempArray.length );
      if( nextArray.length > 0 ) nextArray.splice( 0, nextArray.length );
      var time = 1000.0/fps;
      var trans;

      while( (trans = transArray.pop() ) ) {
         trans.elapsed += time;
         //document.getElementById('txt').value=trans.elapsed;
         if( trans.elapsed >= trans.time ) {
            var tostyle = trans.tostyle;
            for( var key in tostyle) {
               trans.object.style[key] = tostyle[key];
            }

            if( trans.next ) {
               nextArray.push( trans.next );
            }
            if( trans.ondone ) {
               trans.ondone();
            }
         }
         else {
            // save the data
            tempArray.push( trans );
            // perform the transitions
            var tostyle = trans.tostyle;
            var fromstyle = trans.fromstyle;
            var ratio = trans.elapsed/trans.time;
            for( var key in tostyle) {
               var fromValue = fromstyle[key];
               var toValue = getNumValue( tostyle[key] );
               var value = ((toValue - fromValue)*ratio)+fromValue;
               trans.object.style[key] = setNumValue( tostyle[key], value );
            }
         }
      }

      // save a copy of the temp array
      transArray.splice( 0, transArray.length );
      if( tempArray.length > 0 ) {
         transArray = transArray.concat(tempArray);
         setTimeout( progress, time );
      }

      // eventually start new transitions.
      if( nextArray.length > 0 ) {
         for( var count in nextArray ) {
            Nest.addTransition( nextArray[count] );
         }
         nextArray.splice( 0, nextArray.length );
      }
   };

   //========================================================== Public interface.
   // Appends a transition
   //
   // object: An entity of the document that must be transited,
   // time: Duration of the transition
   // tostyle: change into the given style; "text |number| text" indicates styles with textual elements.
   // fromstyle: (optional) Initial status of the transition; use only numbers.
   // next: (optional) Transition to perform when this is complete.
   // ondone: (optional) callback to execute when the transition is complete.

    if (typeof Nest.addTransition !== 'function') {
      Nest.addTransition = function( obj ) {

         // first, update the object as we desire.
         var fromstyle = obj.fromstyle;
         var tostyle = obj.tostyle;
         if( fromstyle ) {
            for( var key in fromstyle) {
               obj.object.style[key] = setNumValue(tostyle[key], fromstyle[key]);
            }
         }

         // reads the current defaults in the object
         fromstyle = {};
         for( var key in tostyle ) {
            fromstyle[key] = getNumFromValue(tostyle[key], getStyle(obj.object, key));
         }
         obj.fromstyle = fromstyle;

         // start the transation
         obj.elapsed = 0;
         if ( transArray.length == 0 ){
            transArray.push( obj );
            setTimeout( progress, 1000.0/fps );
         }
         else {
            transArray.push( obj );
         }
      };
   }

   if (typeof Nest.transite !== 'function') {
      Nest.transite = function( object_, time_, tostyle_, fromstyle_, next_, ondone_ ) {
         Nest.addTransition( {
            object: object_, time: time_, fromstyle: fromstyle_,  tostyle: tostyle_, next: next_, ondone:ondone_
            }
         );
      }
   }

}());

