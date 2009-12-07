/*
   FALCON - The Falcon Programming Language.
   FILE: inspect.cpp

   Deep inspect function.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 15 Jul 2009 07:03:13 -0700

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/coredict.h>
#include <falcon/cclass.h>
#include <falcon/corefunc.h>
#include <falcon/stream.h>
#include <falcon/membuf.h>

namespace Falcon {
namespace core {

void inspect_internal( VMachine *vm, const Item *elem, int32 level, int32 maxLevel, int32 maxSize, Item* i_stream, bool add = true, bool addLine=true );

void inspect_internal( VMachine *vm, const Item *elem, int32 level, int32 maxLevel, int32 maxSize, Item* i_stream, bool add, bool addline )
{

   uint32 count;
   int32 i;
   Stream *stream = i_stream != 0 ?
         dyncast<Stream*>(i_stream->asObjectSafe()->getFalconData()) :
         vm->stdErr();

   // return if we reached the maximum level.
   if ( maxLevel >= 0 && level > maxLevel )
   {
      stream->writeString( "..." );
      if ( addline )
         stream->writeString( "\n" );
      return;
   }

   if ( stream == 0 )
   {
      stream = vm->stdOut();
      if ( stream == 0 )
         return;
   }

   if ( add )
      for ( i = 0; i < level*3; i ++ )
      {
         stream->put( 0x20 ); // space
      }

   if ( elem == 0 ) {
      stream->writeString( "Nothing" );
      if ( addline )
         stream->writeString( "\n" );
      return;
   }

   String temp;

   switch( elem->type() )
   {
      case FLC_ITEM_NIL:
         stream->writeString( "Nil" );
      break;

      case FLC_ITEM_UNB:
            stream->writeString( "_" );
      break;

      case FLC_ITEM_BOOL:
         stream->writeString( elem->asBoolean() ? "true" : "false" );
      break;


      case FLC_ITEM_INT:
         temp.writeNumber( elem->asInteger() );
         stream->writeString( "int(" );
         stream->writeString( temp );
         stream->writeString( ")" );
      break;

      case FLC_ITEM_NUM:
         temp.writeNumber( elem->asNumeric(), "%g" );
         stream->writeString( "num(" );
         stream->writeString( temp );
         stream->writeString( ")" );
      break;

      case FLC_ITEM_RANGE:
         elem->toString(temp);
         stream->writeString( temp );
      break;

      case FLC_ITEM_STRING:
         stream->writeString( "\"" );
         if ( maxSize < 0 || elem->asString()->length() < (uint32) maxSize )
         {
            stream->writeString( *elem->asString() );
            stream->writeString( "\"" );
         }
         else {
            stream->writeString( elem->asString()->subString(0, maxSize ) );
            stream->writeString( " ... \"" );
         }
      break;

      case FLC_ITEM_LBIND:
         stream->writeString( "&" );
         stream->writeString( *elem->asLBind() );
      break;

      case FLC_ITEM_MEMBUF:
      {
         MemBuf *mb = elem->asMemBuf();
         temp = "MemBuf(";
         temp.writeNumber( (int64) mb->length() );
         temp += ",";
         temp.writeNumber( (int64) mb->wordSize() );
         temp += ")";

         if ( maxSize == 0 )
            stream->writeString( temp );
         else {
            temp += " [\n";

            String fmt;
            int limit = 0;
            switch ( mb->wordSize() )
            {
               case 1: fmt = "%02" LLFMT "X"; limit = 24; break;
               case 2: fmt = "%04" LLFMT "X"; limit = 12; break;
               case 3: fmt = "%06" LLFMT "X"; limit = 9; break;
               case 4: fmt = "%08" LLFMT "X"; limit = 6; break;
            }

            int written = 0;
            uint32 max = maxSize < 0 || mb->length() < (uint32) maxSize ? mb->length() : (uint32) maxSize;
            for( count = 0; count < max; count++ )
            {
               temp.writeNumber( (int64)  mb->get( count ), fmt );
               temp += " ";
               written ++;
               if ( written == limit )
               {
                  temp += "\n";
                  written = 0;
               }
               stream->writeString( temp );
               temp = "";
            }
            if ( count == (uint32) maxSize )
               stream->writeString( " ... " );
            stream->writeString( "]" );
         }
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *arr = elem->asArray();
         temp = "Array[";
         temp.writeNumber( (int64) arr->length() );
         temp += "]";
         stream->writeString( temp );

         if ( level == maxLevel )
         {
            stream->writeString( "{...}" );
            break;
         }

         stream->writeString( "{\n" );

         for( count = 0; count < arr->length(); count++ ) {
            inspect_internal( vm, & ((*arr)[count]), level + 1, maxLevel, maxSize, i_stream, true, true );
         }

         for ( i = 0; i < level; i ++ )
         {
            stream->writeString( "   " );
         }
         stream->writeString( "}" );
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *dict = elem->asDict();
         temp = "Dict[";
         temp.writeNumber( (int64) dict->length() );
         temp += "]";
         stream->writeString( temp );

         if ( level == maxLevel )
         {
            stream->writeString( "{...}" );
            break;
         }

         stream->writeString( "{\n" );

         Iterator iter( &dict->items() );
         while( iter.hasCurrent() )
         {
            inspect_internal( vm, &iter.getCurrentKey(), level + 1, maxLevel, maxSize, i_stream, true, false );
            stream->writeString( " => " );
            inspect_internal( vm, &iter.getCurrent(), level + 1, maxLevel, maxSize, i_stream, false, true );
            iter.next();
         }
         for ( i = 0; i < level; i ++ )
         {
            stream->writeString("   ");
         }
         stream->writeString( "}" );
      }
      break;

      case FLC_ITEM_OBJECT:
      {
         CoreObject *arr = elem->asObjectSafe();
         stream->writeString( "Object of class " + arr->generator()->symbol()->name() );
         if ( level == maxLevel )
         {
            stream->writeString( "{...}" );
            break;
         }

         stream->writeString( " {\n" );
         const PropertyTable &pt = arr->generator()->properties();

         for( count = 0; count < pt.added() ; count++ )
         {
            for ( i = 0; i < (level+1); i ++ )
            {
               stream->writeString("   ");
            }
            const String &propName = *pt.getKey( count );
			
			if ( pt.getEntry(count).isWriteOnly() )
			{
				stream->writeString( "(" );
				stream->writeString( propName + ")\n" );
			}
			else 
			{
				stream->writeString( propName + " => " );
				Item dummy;
				arr->getProperty( propName, dummy);
				inspect_internal( vm, &dummy, level + 1, maxLevel, maxSize, i_stream, false, true );
			}
         }
         for ( i = 0; i < level; i ++ )
         {
            stream->writeString("   ");
         }
         stream->writeString( "}" );
      }
      break;

      case FLC_ITEM_CLASS:
         stream->writeString( "Class " + elem->asClass()->symbol()->name() );
      break;

      case FLC_ITEM_METHOD:
      {
         temp = "Method ";
         temp += "->" + elem->asMethodFunc()->name();
         stream->writeString( temp );

         Item itemp;
         elem->getMethodItem( itemp );

         inspect_internal( vm, &itemp, level + 1, maxLevel, maxSize, i_stream, true, true );
         for ( i = 0; i < level; i ++ )
         {
            stream->writeString("   ");
         }
         stream->writeString( "}" );
      }
      break;

      case FLC_ITEM_CLSMETHOD:
         temp = "Cls.Method 0x";
         temp.writeNumberHex( (uint64) elem->asMethodClassOwner() );
         temp += "->" + elem->asMethodClass()->symbol()->name();
         stream->writeString( temp );
      break;

      case FLC_ITEM_FUNC:
      {
         const Symbol *funcSym = elem->asFunction()->symbol();

         if ( funcSym->isExtFunc() )
         {
            stream->writeString( "Ext. Function " + funcSym->name() );
         }
         else {
            stream->writeString( "Function " + funcSym->name() );

            FuncDef *def = funcSym->getFuncDef();
            uint32 itemId = def->onceItemId();
            if ( itemId != FuncDef::NO_STATE )
            {
               if ( elem->asFunction()->liveModule()->globals()[ itemId ].isNil() )
                  stream->writeString( "{ not called }");
               else
                  stream->writeString( "{ called }");
            }
         }
      }
      break;

      case FLC_ITEM_REFERENCE:
         stream->writeString( "Ref to " );
         inspect_internal( vm, elem->dereference(), level + 1, maxLevel, maxSize, i_stream, false, true );
      break;

      default:
         stream->writeString( "Invalid type");
   }
   if ( addline )
      stream->writeString( "\n" );

   stream->flush();
}


/*#
   @function inspect
   @inset core_basic_io
   @param item The item to be inspected.
   @optparam depth Maximum inspect depth.
   @optparam maxLen Limit the display size of possibly very long items as i.e. strings or membufs.
   @optparam stream Different stream where to send the dump.
   @brief Displays the deep contents of an item.

   This is mainly a debugging function that prints all the available
   informations on the item on the auxiliary stream. This function
   should not be used except for testing scripts and checking what
   they put in arrays, dictionaries, objects, classes or simple items.

   Output is sent to the VM auxiliary stream; for stand-alone scripts,
   this translates into the "standard error stream". Embedders may provide
   simple debugging facilities by overloading and intercepting the VM
   auxiliary stream and provide separate output for that.

   This function traverses arrays and items deeply; there isn't any protection
   against circular references, which may cause endless loop. However, the
   default maximum depth is 3, which is a good depth for debugging (goes deep,
   but doesn't dig beyond average interesting points). Set to -1 to have
   infinite depth.

   By default, only the first 60 characters of strings and elements of membufs
   are displayed. You may change this default by providing a @b maxLen parameter.

   You may create personalized inspect functions using forward bindings, like
   the following:
   @code
   compactInspect = .[inspect depth|1 maxLen|15]
   @endcode

   And then, you may inspect a list of item with something like:
   @code
   linsp = .[ dolist _compactInspect x ]
   linsp( ["hello", "world"] )
   @endcode

*/

FALCON_FUNC  inspect ( ::Falcon::VMachine *vm )
{
   Item *i_item = vm->param(0);
   Item *i_depth = vm->param(1);
   Item *i_maxLen = vm->param(2);
   Item *i_stream = vm->param(3);

   if ( i_item == 0
      || ( i_depth != 0 && ! i_depth->isNil() && ! i_depth->isOrdinal() )
      || ( i_maxLen != 0 && ! i_maxLen->isNil() && ! i_maxLen->isOrdinal() )
      || ( i_stream != 0 && ! i_stream->isNil() && ! i_stream->isOfClass("Stream") ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X,[N],[N],[Stream]") );
   }

   int32 depth = (int32) (i_depth == 0 || i_depth->isNil() ? 3 : i_depth->forceInteger());
   int32 maxlen = (int32) (i_maxLen == 0 || i_maxLen->isNil() ? 60 : i_maxLen->forceInteger());

   inspect_internal( vm, i_item, 0, depth, maxlen, i_stream );
}




static void describe_internal( VMachine *vm, String &tgt, const Item *elem, int32 level, int32 maxLevel, int32 maxSize )
{
   uint32 count;

   // return if we reached the maximum level.
   if ( maxLevel >= 0 && level > maxLevel )
   {
      tgt += "...";
      return;
   }

   switch( elem->type() )
   {
      case FLC_ITEM_NIL:
         tgt += "Nil";
      break;

      case FLC_ITEM_UNB:
         tgt += "_";
      break;


      case FLC_ITEM_BOOL:
         tgt += elem->asBoolean() ? "true" : "false";
      break;


      case FLC_ITEM_INT:
         tgt.writeNumber( elem->asInteger() );
      break;

      case FLC_ITEM_NUM:
         tgt.writeNumber( elem->asNumeric(), "%g" );
      break;

      case FLC_ITEM_RANGE:
         elem->toString(tgt);
      break;

      case FLC_ITEM_STRING:
         tgt += "\"";
         if ( maxSize < 0 || elem->asString()->length() < (uint32) maxSize )
         {
            tgt += *elem->asString();
            tgt += "\"";
         }
         else {
            tgt += elem->asString()->subString(0, maxSize );
            tgt += " ... \"";
         }
      break;

      case FLC_ITEM_LBIND:
         tgt += "&";
         tgt += *elem->asLBind();
         if (elem->isFutureBind())
         {
            tgt +="|";
            describe_internal( vm, tgt, &elem->asFutureBind(), level+1, maxLevel, maxSize );
         }
      break;

      case FLC_ITEM_MEMBUF:
      {
         MemBuf *mb = elem->asMemBuf();
         tgt += "MB(";
         tgt.writeNumber( (int64) mb->length() );
         tgt += ",";
         tgt.writeNumber( (int64) mb->wordSize() );
         tgt += ")";

         tgt += " [";

         String fmt;
         int limit = 0;
         switch ( mb->wordSize() )
         {
            case 1: fmt = "%02" LLFMT "X"; limit = 24; break;
            case 2: fmt = "%04" LLFMT "X"; limit = 12; break;
            case 3: fmt = "%06" LLFMT "X"; limit = 9; break;
            case 4: fmt = "%08" LLFMT "X"; limit = 6; break;
         }

         uint32 max = maxSize < 0 || mb->length() < (uint32) maxSize ? mb->length() : (uint32) maxSize;
         for( count = 0; count < max; count++ )
         {
            tgt.writeNumber( (int64)  mb->get( count ), fmt );
            tgt += " ";
         }
         if ( count == (uint32) maxSize )
            tgt += " ...";
         tgt += "]";
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *arr = elem->asArray();
         tgt += "[";

         if ( level == maxLevel )
         {
            tgt += "...]";
            break;
         }

         for( count = 0; count < arr->length(); count++ ) {
            if ( count == 0 ) tgt += " ";

            describe_internal( vm, tgt, & ((*arr)[count]), level + 1, maxLevel, maxSize );

            if ( count + 1 < arr->length() )
               tgt += ", ";
         }

         tgt +="]";
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *dict = elem->asDict();
         if( dict->isBlessed() )
            tgt += "*";

         tgt += "[";

         if ( level == maxLevel )
         {
            tgt += "...=>...]";
            break;
         }

         if ( dict->length() == 0 )
         {
            tgt += "=>]";
            break;
         }

         Item key, value;
         Iterator iter( &dict->items() );

         // separate the first loop to be able to add ", "
         describe_internal( vm, tgt, &iter.getCurrentKey(), level + 1, maxLevel, maxSize );
         tgt += " => ";
         describe_internal( vm, tgt, &iter.getCurrent(), level + 1, maxLevel, maxSize );
         iter.next();
         while( iter.hasCurrent() )
         {
            tgt += ", ";
            describe_internal( vm, tgt, &iter.getCurrentKey(), level + 1, maxLevel, maxSize );
            tgt += " => ";
            describe_internal( vm, tgt, &iter.getCurrent(), level + 1, maxLevel, maxSize );
            iter.next();
         }

         tgt += "]";
      }
      break;

      case FLC_ITEM_OBJECT:
      {
         CoreObject *arr = elem->asObjectSafe();
         tgt += arr->generator()->symbol()->name() + "(){ ";

         if ( level == maxLevel )
         {
            tgt += "...}";
            break;
         }

         const PropertyTable &pt = arr->generator()->properties();

         for( count = 0; count < pt.added() ; count++ )
         {
            const String &propName = *pt.getKey( count );
			
			// write only?
			if ( pt.getEntry( count ).isWriteOnly() )
			{
				tgt.A( "(" ).A( propName ).A(")");
			}
			else
			{
				Item dummy;
				arr->getProperty( propName, dummy );

				// in describe skip methods.
				if ( dummy.isFunction() || dummy.isMethod() )
				   continue;
				
				tgt += propName + " = ";

				describe_internal( vm, tgt, &dummy, level + 1, maxLevel, maxSize );
			}

            if (count+1 < pt.added())
            {
               tgt += ", ";
            }
         }

         tgt += "}";
      }
      break;

      case FLC_ITEM_CLASS:
         tgt += "Class " + elem->asClass()->symbol()->name();
      break;

      case FLC_ITEM_METHOD:
      {
         tgt += "(";
         Item itemp;
         elem->getMethodItem( itemp );
         describe_internal( vm, tgt, &itemp, level + 1, maxLevel, maxSize );
         tgt += ").";
         tgt += elem->asMethodFunc()->name() + "()";
      }
      break;

      case FLC_ITEM_CLSMETHOD:
         tgt += "Class ";
         tgt += elem->asMethodClassOwner()->generator()->symbol()->name();
         tgt += "." + elem->asMethodClass()->symbol()->name() + "()";
      break;

      case FLC_ITEM_FUNC:
      {
         const Symbol *funcSym = elem->asFunction()->symbol();
         tgt += funcSym->name() + "()";
      }
      break;

      case FLC_ITEM_REFERENCE:
         tgt += "->";
         describe_internal( vm, tgt, elem->dereference(), level + 1, maxLevel, maxSize );
      break;

      default:
         tgt += "?";
   }
}


/*#
   @function describe
   @param item The item to be inspected.
   @optparam depth Maximum inspect depth.
   @optparam maxLen Limit the display size of possibly very long items as i.e. strings or membufs.
   @brief Returns the deep contents of an item on a string representation.

   This function returns a string containing a representation of the given item.
   If the item is deep (an array, an instance, a dictionary) the contents are
   also passed through this function.

   This function traverses arrays and items deeply; there isn't any protection
   against circular references, which may cause endless loop. However, the
   default maximum depth is 3, which is a good depth for debugging (goes deep,
   but doesn't dig beyond average interesting points). Set to -1 to have
   infinite depth.

   By default, only the first 60 characters of strings and elements of membufs
   are displayed. You may change this default by providing a @b maxLen parameter.

   You may create personalized inspect functions using forward bindings, like
   the following:
   @code
   compactDescribe = .[inspect depth|1 maxLen|15]
   @endcode
*/

/*#
   @method describe BOM
   @optparam depth Maximum inspect depth.
   @optparam maxLen Limit the display size of possibly very long items as i.e. strings or membufs.
   @brief Returns the deep contents of an item on a string representation.

   This method returns a string containing a representation of this item.
   If the item is deep (an array, an instance, a dictionary) the contents are
   also passed through this function.

   This method traverses arrays and items deeply; there isn't any protection
   against circular references, which may cause endless loop. However, the
   default maximum depth is 3, which is a good depth for debugging (goes deep,
   but doesn't dig beyond average interesting points). Set to -1 to have
   infinite depth.

   By default, only the first 60 characters of strings and elements of membufs
   are displayed. You may change this default by providing a @b maxLen parameter.
*/

FALCON_FUNC  mth_describe ( ::Falcon::VMachine *vm )
{
   Item *i_item;
   Item *i_depth;
   Item *i_maxLen;

   if( vm->self().isMethodic() )
   {
      i_item = &vm->self();
      i_depth = vm->param(0);
      i_maxLen = vm->param(1);
   }
   else {
      i_item = vm->param(0);
      i_depth = vm->param(1);
      i_maxLen = vm->param(2);
   }

   if ( i_item == 0
      || ( i_depth != 0 && ! i_depth->isNil() && ! i_depth->isOrdinal() )
      || ( i_maxLen != 0 && ! i_maxLen->isNil() && ! i_maxLen->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->self().isMethodic() ? "[N],[N]" : "X,[N],[N]") );
   }

   int32 depth = (int32) (i_depth == 0 || i_depth->isNil() ? 3 : i_depth->forceInteger());
   int32 maxlen = (int32) (i_maxLen == 0 || i_maxLen->isNil() ? 60 : i_maxLen->forceInteger());


   String temp;
   describe_internal( vm, temp, i_item, 0, depth, maxlen );
   CoreString* res = new CoreString(temp);
   vm->retval( res );
}


}}

/* end of inspect.cpp */
