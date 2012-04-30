/* Falcon - Nest framework - AJAX support */

var Nest;
if(!Nest) { Nest = {}; }

(function () {
   "use strict";

if (!Array.prototype.indexOf) {
    Array.prototype.indexOf = function (obj, fromIndex) {
        if (fromIndex == null) {
            fromIndex = 0;
        } else if (fromIndex < 0) {
            fromIndex = Math.max(0, this.length + fromIndex);
        }
        for (var i = fromIndex, j = this.length; i < j; i++) {
            if (this[i] === obj)
                return i;
        }
        return -1;
    };
}

   //============================================================================ Private part
   function ajax( url, data, callback ) {
      var http = window.XMLHttpRequest ? new XMLHttpRequest() : new ActiveXObject("Microsoft.XMLHTTP");

      var params = "";
      if(data) {
         /*
         for( var key in data )
         {
            if ( params != "" ) { params = params + "&"; }
            params = params + encodeURIComponent(key) + "=" + encodeURIComponent(data[key]);
         }
         */
         params = "params=" + encodeURIComponent( JSON.stringify( data ) );

         http.open("POST", url, true);

         //Send the proper header information along with the request
         http.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
         // http.setRequestHeader("Content-length", params.length);
         // http.setRequestHeader("Connection", "close");
      }
      else {
         http.open("GET", url, true);
      }

      http.onreadystatechange = function() { //Call a function when the state changes.
         if(http.readyState == 4 ) {
            if ( http.status == 200 ) {
               var obj;
               try {
                  // try to understand as json.
                  obj = JSON.parse(http.responseText);
               }
               catch( err ) {
                  // if not json, raise proper error.
                  Nest.onJSONError( http.responseText, err )
               }

               if( obj ) {
                  // application error?
                  if ( obj.error ) { Nest.onAPIError( obj ); }
                  else if (callback) { callback( obj ); }
               }
            }
            else {
               Nest.onAJAXError( http.status, http.responseText );
            }
         }
      }

      if( data ) {
         http.send(params);
      }
      else {
         http.send(null);
      }
   }

   // prepare the information to be sent.
   function prepareInfos( widID, obj, infosToSend ) {
      var i = 0;
      var element;
      while( i < infosToSend.length ) {
         var info = infosToSend[i];
         var useName = false;
         if( info.substring( 0, 1 ) == '*' )
         {
            useName = true;
            info = info.substring(1);
         }
         var idArr = info.split("/");
         var valname = idArr.pop();

         if( idArr.length == 0 ) {
            element = document.getElementById( widID );
         }
         else {
            if( idArr[0] == "" ) {
               // the path is absolute
               idArr.shift();
               element = document.getElementById( idArr.join(".") );
            }
            else {
               var widIdArr = widID.split( "." );
               while( idArr.length > 0 && idArr[0] == '..' ) {
                  widIdArr.pop();
                  idArr.shift();
               }
               widIdArr = widIdArr.concat( idArr );
               var wid = widIdArr.join( "." );
               element = document.getElementById( wid );
            }
         }
         
         if (element != null) {
            // get the proper value
            var value;
            if ( valname == 'value' && element.getValue != null )
                value = element.getValue();
            else
               value = element[valname];

            // recreate the full entity name, re-localized after .. purging.
            
            var fieldName;
            if( useName ) {
               fieldName = element.name;               
            }
            if ( ! useName || fieldName == null )
            {
               idArr.push( valname );
               fieldName = idArr.join(".");
            }
            
            // save the field
            obj[fieldName.replace("[]", "")] = value;
         }

         i = i + 1;
      }
   }

   // Handler for set message
   function handler_set( obj ) {
      var element = document.getElementById( obj.id );
      if( element ) {
         if( obj.property == 'value' && element.setValue ) {
            element.setValue( obj.value )
         }
         else {
            element[obj.property] = obj.value;
         }
      }
   }

   // Handler for set message
   function handler_set_style( obj ) {
      var element = document.getElementById( obj.id );
      if( element ) { element.style[obj.property] = obj.value; }
   }
   
   // Handler for invoke message
   function handler_invoke( obj ) {
      var element = document.getElementById( obj.id );
      if( element ) { element[obj.method].call( element, obj.param ); }
   }

   function prepareInit( widID, obj ) {
      // get the root widget.
      
      var list = widID.split(".");
      while( list.length > 0 )
      {
         widID = list.join(".");
         var rootElement = document.getElementById( widID );
         // If we have a root widget, see if we assigned an init info to it.
         if ( rootElement && rootElement.Nest_initInfo )
         {
            obj.init = rootElement.Nest_initInfo;
            break;
         }
         list.pop();
      }
   }

   //=================================================================================== Public interace
   // Method 'i' -- shortcut for document.getElementByID
   if (typeof Nest.i !== 'function') {
      Nest.i = function ( id ) { return document.getElementById( id ); }
   }

   // Stop event propagation
   Nest.eatEvent = function(evt){
      if(typeof(event) != "undefined") event.cancelBubble = true;
      else evt.stopPropagation();
   }
   
   // All the widgets declared by nest
   if (!Nest.w) { Nest.w = new Array();}

   // Method 'rw' -- relative widget.
   if (typeof Nest.rw !== 'function') {
      Nest.rw = function ( wid, path ) {
         var pathArr = path.split("/")
         var widIdArr = wid.id.split( "." );
         while( pathArr.length > 0 && pathArr[0] == '..' ) {
            widIdArr.pop();
            pathArr.shift();
         }
         widIdArr = widIdArr.concat( pathArr );
         var widID = widIdArr.join( "." );
         wid = document.getElementById( widID );
         return wid;
      }
   }
   
   // Method 'ajax'
   if (typeof Nest.ajax !== 'function') {
      Nest.ajax = function ( req_id, params, callback ) {
         var url = "./?a=" + req_id;
         ajax( url, params, callback );
      }
   }

   if (typeof Nest.setWidVal !== 'function') {
      Nest.setWidVal = function ( wid, value ) {
         var element = Nest.i(wid);
         if( element != null ) {
            if( element.setValue != null ) {
               element.setValue( value );
            }
            else {
               element.value = value;
            }
         }         
      }
   }

   // Method 'widgetMsg' -- Sending AJAX requests to remote widget server.
   if (typeof Nest.widgetAJAX !== 'function') {
      Nest.widgetAJAX = function ( widClass, widID, msg, infosToSend, extraParams ) {
         // the object to be sent.
         var objToSend = { 'widget': widClass, 'id': widID, 'msg': msg };
         // let's get rid of the details now -- this is extra data to send as-is
         if( extraParams != null ) {objToSend.extra = extraParams;}
         prepareInit( widID, objToSend );
         
         // read the infos and store them away
         if( infosToSend != null) {
            var infos = {};
            prepareInfos( widID, infos, infosToSend );
            objToSend["infos"] = infos;
         }

         var url = "./?w=" + widClass;
         //alert( JSON.stringify( objToSend ) );
         ajax( url, objToSend, Nest.widgetUpdate );
      }
   }

   // Method 'message' -- sends a local message to listeners in the page.
   if (typeof Nest.message !== 'function') {
      Nest.message = function ( wid, msg, value ) {
         var listener = Nest.listeners[wid.id];
         if( listener ) {
            for (var i = 0; i < listener.length; i++) {
               var lrec = listener[i];
               var func = lrec.func;
               var tgtid = lrec.tgt;
               func.call( tgtid, wid, msg, value );
            }
         }
      }
   }

   // Method 'listen' -- Waits for updates on a certain widget ID
   // callbacks are in this prototype: func( target, source_wid, msg, value );
   if (typeof Nest.listen !== 'function') {
      Nest.listen = function ( target, wid, cbfunc ) {
         var listener = Nest.listeners[wid];
         var listenRecord = { "tgt": target, "func": cbfunc };
         if( listener ) {
            listener.push( listenRecord );
         }
         else {
            Nest.listeners[wid] = Array( listenRecord );
         }
      }
   }   
   
   
   // Method 'widgetUpdate' -- handling requests from widget server.
   if (typeof Nest.widgetUpdate !== 'function') {
      Nest.widgetUpdate = function ( obj ) {
         // handle multiple messages.
         if( typeof obj == 'object' && obj.length ) {
            var i = 0;
            while( i < obj.length ) { Nest.processMessage( obj[i] ); i = i + 1; }
         }
         else {
            Nest.processMessage( obj );
         }
      }
   }

   // Method 'processMessage' -- handling a single request from widget server.
   if (typeof Nest.processMessage !== 'function') {
      Nest.processMessage = function ( obj ) {
         if( obj.message ) {
            var handler = Nest.messageHandlers[ obj.message ];
            if( ! handler ) {
               Nest.onMessageNotFound( obj );
            }
            else {
               handler.method.call( handler.object, obj );
            }
         }
         else {
            Nest.onWidgetUpdateError( obj );
         }
      }
   }

   // Method 'processMessage' -- handling a single request from widget server.
   if (typeof Nest.listenAJAX !== 'function') {
      Nest.listenAJAX = function ( msg, obj, func ) {        
         Nest.messageHandlers[ msg ] = { object: obj, method: func };
      }
   }
   
   //=========================================================================== Error management.
   if (typeof Nest.onAJAXError !== 'function') {
      Nest.onAJAXError = function( code, text ){
         alert( "Nest framework AJAX error.\n" +
            "Response from server: " + code + "\n" +
            text );
      }
   }

   if (typeof Nest.onJSONError !== 'function') {
      Nest.onJSONError = function( text, synerr ){
         alert( "Nest framework AJAX error.\n" +
            "JSON parse error: " + synerr.name+"\n"+ synerr.message +"\n" +
            "Response was: " + text
            );
      }
   }

   if (typeof Nest.onAPIError !== 'function') {
      Nest.onAPIError = function( obj ){
         alert( "Nest framework AJAX API error.\n" +
            "Remote API error : " + obj.error +"\n"+
               obj.errorDesc
            );
      }
   }

    if (typeof Nest.onMessageNotFound !== 'function') {
      Nest.onMessageNotFound = function( obj ){
         alert( "No handler registered for Nest widget message '" + obj.message + "'.\n" +
            "Received: " + obj +"\n"
            );
      }
   }

    if (typeof Nest.onWidgetUpdateError !== 'function') {
      Nest.onWidgetUpdateError = function( obj ) {
         alert( "Not a widget update message in Nest widget update.\n" +
            "Received: " + obj +"\n"
            );
      }
   }

   //=========================================================================== Object initialization.
   // set the default widget server message handlers
   if (! Nest.messageHandlers ) {
      Nest.messageHandlers = {
         'set': { object: null, method: handler_set},
         'set_style': { object: null, method: handler_set_style},
         'invoke': { object: null, method: handler_invoke}
      }
   }

   // Set the local message handlers
   if (! Nest.listeners ) {
      Nest.listeners = {};
   }

}());


/* http://www.JSON.org/json2.js -- See http://www.JSON.org/js.html */
var JSON;
if (!JSON) {
    JSON = {};
}
(function () {
    "use strict";

    function f(n) {
        // Format integers to have at least two digits.
        return n < 10 ? '0' + n : n;
    }

    if (typeof Date.prototype.toJSON !== 'function') {

        Date.prototype.toJSON = function (key) {

            return isFinite(this.valueOf()) ?
                this.getUTCFullYear() + '-' +
                f(this.getUTCMonth() + 1) + '-' +
                f(this.getUTCDate()) + 'T' +
                f(this.getUTCHours()) + ':' +
                f(this.getUTCMinutes()) + ':' +
                f(this.getUTCSeconds()) + 'Z' : null;
        };

        String.prototype.toJSON =
            Number.prototype.toJSON =
            Boolean.prototype.toJSON = function (key) {
                return this.valueOf();
            };
    }

    var cx = /[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,
        escapable = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,
        gap,
        indent,
        meta = { // table of character substitutions
            '\b': '\\b',
            '\t': '\\t',
            '\n': '\\n',
            '\f': '\\f',
            '\r': '\\r',
            '"' : '\\"',
            '\\': '\\\\'
        },
        rep;


    function quote(string) {
        escapable.lastIndex = 0;
        return escapable.test(string) ? '"' + string.replace(escapable, function (a) {
            var c = meta[a];
            return typeof c === 'string' ? c :
                '\\u' + ('0000' + a.charCodeAt(0).toString(16)).slice(-4);
        }) + '"' : '"' + string + '"';
    }


    function str(key, holder) {
        var i, // The loop counter.
            k, // The member key.
            v, // The member value.
            length,
            mind = gap,
            partial,
            value = holder[key];
        if (value && typeof value === 'object' &&
                typeof value.toJSON === 'function') {
            value = value.toJSON(key);
        }
        if (typeof rep === 'function') {
            value = rep.call(holder, key, value);
        }
        switch (typeof value) {
        case 'string':
            return quote(value);

        case 'number':
            return isFinite(value) ? String(value) : 'null';

        case 'boolean':
        case 'null':
            return String(value);
        case 'object':
            if (!value) {
                return 'null';
            }
            gap += indent;
            partial = [];
            if (Object.prototype.toString.apply(value) === '[object Array]') {
                length = value.length;
                for (i = 0; i < length; i += 1) {
                    partial[i] = str(i, value) || 'null';
                }
                v = partial.length === 0 ? '[]' : gap ?
                    '[\n' + gap + partial.join(',\n' + gap) + '\n' + mind + ']' :
                    '[' + partial.join(',') + ']';
                gap = mind;
                return v;
            }

            if (rep && typeof rep === 'object') {
                length = rep.length;
                for (i = 0; i < length; i += 1) {
                    if (typeof rep[i] === 'string') {
                        k = rep[i];
                        v = str(k, value);
                        if (v) {
                            partial.push(quote(k) + (gap ? ': ' : ':') + v);
                        }
                    }
                }
            } else {
                for (k in value) {
                    if (Object.prototype.hasOwnProperty.call(value, k)) {
                        v = str(k, value);
                        if (v) {
                            partial.push(quote(k) + (gap ? ': ' : ':') + v);
                        }
                    }
                }
            }
            v = partial.length === 0 ? '{}' : gap ?
                '{\n' + gap + partial.join(',\n' + gap) + '\n' + mind + '}' :
                '{' + partial.join(',') + '}';
            gap = mind;
            return v;
        }
    }
    if (typeof JSON.stringify !== 'function') {
        JSON.stringify = function (value, replacer, space) {
            var i;
            gap = '';
            indent = '';
            if (typeof space === 'number') {
                for (i = 0; i < space; i += 1) {
                    indent += ' ';
                }
            } else if (typeof space === 'string') {
                indent = space;
            }
            rep = replacer;
            if (replacer && typeof replacer !== 'function' &&
                    (typeof replacer !== 'object' ||
                    typeof replacer.length !== 'number')) {
                throw new Error('JSON.stringify');
            }
            return str('', {'': value});
        };
    }
    if (typeof JSON.parse !== 'function') {
        JSON.parse = function (text, reviver) {
            var j;

            function walk(holder, key) {
                var k, v, value = holder[key];
                if (value && typeof value === 'object') {
                    for (k in value) {
                        if (Object.prototype.hasOwnProperty.call(value, k)) {
                            v = walk(value, k);
                            if (v !== undefined) {
                                value[k] = v;
                            } else {
                                delete value[k];
                            }
                        }
                    }
                }
                return reviver.call(holder, key, value);
            }
            text = String(text);
            cx.lastIndex = 0;
            if (cx.test(text)) {
                text = text.replace(cx, function (a) {
                    return '\\u' +
                        ('0000' + a.charCodeAt(0).toString(16)).slice(-4);
                });
            }
            if (/^[\],:{}\s]*$/
                    .test(text.replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, '@')
                        .replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, ']')
                        .replace(/(?:^|:|,)(?:\s*\[)+/g, ''))) {
                j = eval('(' + text + ')');
                return typeof reviver === 'function' ?
                    walk({'': j}, '') : j;
            }
            throw new SyntaxError('JSON.parse');
        };
    }
}());

