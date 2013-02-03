/*
   FALCON - The Falcon Programming Language.
   FILE: classre.cpp

   RE2 object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Feb 2013 13:49:49 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classre.cpp"

#include <falcon/classes/classre.h>
#include <falcon/range.h>
#include <falcon/itemid.h>
#include <falcon/itemdict.h>
#include <falcon/itemarray.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>
#include <falcon/errors/accesserror.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/operanderror.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>

#include "../re2/re2/re2.h"

#include <map>

#define MAX_CAPTURE_COUNT 36

namespace Falcon {

//
// Class properties used for enumeration
//

ClassRE::ClassRE():
   ClassUser( "RE", FLC_CLASS_ID_RE ),
   FALCON_INIT_PROPERTY( captures ),
   FALCON_INIT_PROPERTY( caseSensitive ),
   FALCON_INIT_PROPERTY( groupNames ),

   FALCON_INIT_METHOD(match),
   FALCON_INIT_METHOD(find),
   FALCON_INIT_METHOD(capture),

   FALCON_INIT_METHOD(replace),
   FALCON_INIT_METHOD(replaceAll),
   FALCON_INIT_METHOD(change),
   FALCON_INIT_METHOD(changeAll),

   FALCON_INIT_METHOD(consume),
   FALCON_INIT_METHOD(consumeMatch)
{
}


ClassRE::~ClassRE()
{
}


int64 ClassRE::occupiedMemory( void* instance ) const
{
   re2::RE2* re2 = static_cast<re2::RE2*>( instance );
   // TODO: REASONABILY precise measurement.
   return sizeof(re2::RE2) + re2->ProgramSize() + 32;
}


void ClassRE::dispose( void* self ) const
{
   delete static_cast<re2::RE2*>( self );
}


void* ClassRE::clone( void* instance ) const
{
   re2::RE2* re2 = static_cast<re2::RE2*>( instance );
   re2::RE2* copy = new re2::RE2( re2->pattern(), re2->options() );

   return copy;
}

void* ClassRE::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

void ClassRE::store( VMContext*, DataWriter* stream, void* instance ) const
{
   re2::RE2* re2 = static_cast<re2::RE2*>( instance );
   TRACE2( "ClassRE::store -- \"%s\"", re2->pattern().c_str() );
   String pattern;
   pattern.fromUTF8(re2->pattern().c_str());
   stream->write( pattern );
   stream->write( re2->options().case_sensitive() );
   stream->write( re2->options().longest_match() );
   stream->write( re2->options().one_line() );
   stream->write( re2->options().never_nl() );

}


void ClassRE::restore( VMContext* ctx, DataReader* dr ) const
{
   static Class* classRe = Engine::instance()->reClass();

   String pattern;
   bool cs;
   bool longest;
   bool one_line;
   bool never_nL;

   dr->read( pattern );
   dr->read( cs );
   dr->read( longest );
   dr->read( one_line );
   dr->read( never_nL );

   TRACE2( "ClassRE::restore -- \"%s\"", pattern.c_ize() );
   re2::RE2::Options opts;
   opts.set_case_sensitive(cs);
   opts.set_longest_match(longest);
   opts.set_one_line(one_line);
   opts.set_never_nl(never_nL);

   re2::RE2* re2 = new re2::RE2(pattern, opts);
   // let's trust our source.
   ctx->pushData( Item( classRe, re2 ) );
}


void ClassRE::describe( void* instance, String& target, int, int ) const
{
   re2::RE2* self = static_cast<re2::RE2*>( instance );

   String temp;
   temp.fromUTF8( self->pattern().c_str() );
   temp.escapeQuotes();
   if( self->options().case_sensitive() )
   {
      target = "r";
   }
   else {
      target = "R";
   }

   target += "'" + temp + "'";

   if( self->options().one_line() )
   {
      target += "o";
   }

   if( self->options().longest_match() )
   {
      target += "l";
   }

   if( self->options().never_nl() )
   {
      target += "n";
   }

}

void ClassRE::gcMarkInstance( void* instance, uint32 mark ) const
{
   static_cast<re2::RE2*>( instance )->gcMark( mark );
}


bool ClassRE::gcCheckInstance( void* instance, uint32 mark ) const
{
   return static_cast<re2::RE2*>( instance )->currentMark() >= mark;
}

//=======================================================================
// Operands
//

bool ClassRE::op_init( VMContext* ctx, void*, int pcount ) const
{
   // no param?
   String* sopts = 0;
   if( pcount == 0 )
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC)
                              .extra("S,[S]"));
   }

   Item* params = ctx->opcodeParams(pcount);

   if( pcount >= 2 )
   {
      Item& i_opts = params[1];
      if( !(i_opts.isString() || i_opts.isNil()) )
      {
         throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC)
                        .extra("S,[S]"));
      }
      else if( i_opts.isString() )
      {
         sopts = i_opts.asString();
      }
   }

   Item& i_pattern = params[0];
   if ( ! i_pattern.isString())
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC)
                              .extra("S,[S]"));
   }
   
   String* pattern = i_pattern.asString();
   re2::RE2::Options opts;

   if( sopts != 0 )
   {
      for( uint32 i = 0; i< sopts->length(); ++i )
      {
         uint32 chr = sopts->getCharAt(i);
         switch( chr )
         {
         case 'i': opts.set_case_sensitive(false); break;
         case 'n': opts.set_never_nl(true); break;
         case 'l': opts.set_longest_match(true); break;
         case 'o': opts.set_one_line(true); break;
         default:
            throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC)
                           .extra("Unrecognized options: "+*sopts));
         }
      }
   }

   re2::RE2* re = new re2::RE2( *pattern, opts );

   if( ! re->ok() )
   {
      String error;
      String temp;

      error.fromUTF8( re->error().c_str() );
      temp.fromUTF8( re->error_arg().c_str() );
      error += " at ";
      error += temp;
      delete re;

      throw new ParamError(ErrorParam(e_regex_def, __LINE__, SRC)
                    .extra(error));
   }

   ctx->opcodeParam(pcount).setUser( this, re );

   return false;
}


static void internal_match( VMContext* ctx, void* instance, bool partial )
{
   re2::RE2* re = static_cast<re2::RE2*>( instance );

   if( ! ctx->topData().isString() )
   {
     throw new OperandError(ErrorParam(e_invalid_op, __LINE__, SRC )
              .extra("S"));
   }

   String* cfr = ctx->topData().asString();
   re2::StringPiece text(*cfr);
   bool match = re->Match(text, 0, text.size(),
                partial ? re2::RE2::UNANCHORED : re2::RE2::ANCHOR_BOTH,
                0, 0);

   ctx->popData();
   ctx->topData().setBoolean( match );
}

void ClassRE::op_div( VMContext* ctx, void* instance ) const
{
   internal_match( ctx, instance, true );
}

static ItemArray* internal_grab( String* target, void * instance, bool grabAll )
{
   re2::RE2* re = static_cast<re2::RE2*>( instance );

   re2::StringPiece captured[MAX_CAPTURE_COUNT];
   int cc = re->NumberOfCapturingGroups() + 1;
   // paranoid...
   if( cc > MAX_CAPTURE_COUNT )
   {
      cc = MAX_CAPTURE_COUNT;
   }

   re2::StringPiece text(*target);

   bool match = re->Match(text, 0, text.size(),
              re2::RE2::UNANCHORED,
              captured, cc);

   if( match )
   {
      ItemArray* capt = new ItemArray;
      capt->reserve( cc );

      // skip the global match
      for( int i = grabAll ? 0 : 1; i < cc; ++i )
      {
         std::string result = captured[i].ToString();
         String* scapt = new String( result.c_str(), result.size() );
         capt->append( FALCON_GC_HANDLE( scapt ) );
      }
      return capt;
   }

   return 0;
}

void ClassRE::op_mod( VMContext* ctx, void* instance ) const
{

   if( ! ctx->topData().isString() )
   {
     throw new OperandError(ErrorParam(e_invalid_op, __LINE__, SRC )
              .extra("S"));
   }

   Item ret;
   String* cfr = ctx->topData().asString();
   ItemArray* capt = internal_grab( cfr, instance, false);
   if( capt != 0 )
   {
      ret = FALCON_GC_HANDLE( capt );
   }

   ctx->popData();
   ctx->topData() = ret;
}

void ClassRE::op_mul( VMContext* ctx, void* instance ) const
{
   internal_match( ctx, instance, false );
}

void ClassRE::op_pow( VMContext* ctx, void* instance ) const
{
   re2::RE2* re = static_cast<re2::RE2*>( instance );

   if( ! ctx->topData().isString() )
   {
     throw new OperandError(ErrorParam(e_invalid_op, __LINE__, SRC )
              .extra("S"));
   }

   String* cfr = ctx->topData().asString();
   re2::StringPiece piece;

   re2::StringPiece text(*cfr);
   bool match = re->Match(text,
              0,
              text.size(),
              re2::RE2::UNANCHORED,
              &piece,
              1);

   Item ret;
   if( match )
   {
      std::string result = piece.ToString();
      String *captured = new String( result.c_str(), result.size() );
      ret = FALCON_GC_HANDLE( captured );
   }

   ctx->popData();
   ctx->topData() = ret;
}


//=====================================================================
// Properties
//

FALCON_DEFINE_PROPERTY_GET(ClassRE, captures)( void* instance, Item& value )
{
   value.setInteger( static_cast<re2::RE2*>( instance )->NumberOfCapturingGroups() );
}

FALCON_DEFINE_PROPERTY_SET(ClassRE, captures)( void* , const Item&  )
{
   throw readOnlyError();
}

FALCON_DEFINE_PROPERTY_GET(ClassRE, groupNames)( void* instance, Item& value )
{
   re2::RE2* re = static_cast<re2::RE2*>( instance );
   const std::map<std::string, int>& names = re->NamedCapturingGroups();
   const std::map<std::string, int>::const_iterator iter = names.begin();
   const std::map<std::string, int>::const_iterator end = names.end();

   ItemDict* dict = new ItemDict;

   while( iter != end )
   {
      String* name = new String;
      name->fromUTF8( iter->first.c_str() );
      int64 pos = iter->second;
      dict->insert( FALCON_GC_HANDLE(name), pos );
   }

   value = FALCON_GC_HANDLE(dict);
}

FALCON_DEFINE_PROPERTY_SET(ClassRE, groupNames)( void* , const Item&  )
{
   throw readOnlyError();
}


FALCON_DEFINE_PROPERTY_GET(ClassRE, caseSensitive)( void* instance, Item& value )
{
   re2::RE2* re = static_cast<re2::RE2*>( instance );
   value.setBoolean(re->options().case_sensitive());
}

FALCON_DEFINE_PROPERTY_SET(ClassRE, caseSensitive)( void*, const Item& )
{
   throw readOnlyError();
}


FALCON_DEFINE_METHOD_P1( ClassRE, match )
{
   Item* i_target = ctx->param(0);

   if( i_target == 0|| ! i_target->isString() )
   {
      throw paramError();
   }

   re2::RE2* re = static_cast<re2::RE2*>(ctx->self().asInst());
   String* target = i_target->asString();
   re2::StringPiece text(*target);

   bool match = re->Match(text, 0, text.size(),
              re2::RE2::UNANCHORED,
              0, 0);

   ctx->returnFrame( Item().setBoolean(match) );
}


FALCON_DEFINE_METHOD_P1( ClassRE, find )
{
   Item* i_target = ctx->param(0);

   if( i_target == 0|| ! i_target->isString() )
   {
      throw paramError();
   }

   re2::RE2* re = static_cast<re2::RE2*>(ctx->self().asInst());
   String* target = i_target->asString();
   re2::StringPiece text(*target);

   re2::StringPiece captured;
   bool match = re->Match(text, 0, text.size(),
              re2::RE2::UNANCHORED,
              &captured, 1);

   if( match )
   {
      String* ret = new String;
      ret->fromUTF8(captured.ToString().c_str());
      ctx->returnFrame(FALCON_GC_HANDLE(ret));
   }
   else {
      ctx->returnFrame();
   }
}


FALCON_DEFINE_METHOD_P1( ClassRE, capture )
{
   Item* i_target = ctx->param(0);
   Item* i_getall = ctx->param(1);

   if( i_target == 0|| ! i_target->isString() )
   {
      throw paramError();
   }

   bool bGetAll = i_getall != 0 ? i_getall->isTrue() : false;
   ItemArray* capt = internal_grab(i_target->asString(), ctx->self().asInst(), bGetAll );
   if( capt != 0 )
   {
      ctx->returnFrame( FALCON_GC_HANDLE(capt) );
   }
   else
   {
      ctx->returnFrame();
   }
}


static bool internal_change( VMContext* ctx, int mode )
{
   Item* i_target = ctx->param(0);
   Item* i_replacer = ctx->param(1);

   if( i_target == 0|| ! i_target->isString() || i_replacer == 0 || ! i_replacer->isString() )
   {
      return false;
   }

   re2::RE2* re = static_cast<re2::RE2*>(ctx->self().asInst());
   String* target = i_target->asString();
   String* replacer = i_replacer->asString();

   String* ret = 0;
   bool result;
   switch( mode )
   {
   case 0: // replace
      ret = new String;
      result = re2::RE2::Extract(*target, *re, *replacer, *ret );
      break;

   case 1: // replace All
      ret = new String(*target);
      result = re2::RE2::GlobalReplace(*ret, *re, *replacer );
      break;

   case 2: // change
      result = re2::RE2::Replace(*target, *re, *replacer );
      break;

   case 3: // replace All
      result = re2::RE2::GlobalReplace(*target, *re, *replacer );
      break;
   }

   if( result )
   {
      if( ret != 0 )
      {
         ctx->returnFrame(FALCON_GC_HANDLE(ret));
      }
      else {
         ctx->returnFrame(Item().setBoolean(true));
      }
   }
   else
   {
      delete ret;
      ctx->returnFrame();
   }

   return true;
}

FALCON_DEFINE_METHOD_P1( ClassRE, replace )
{
   if( ! internal_change( ctx, 0 ) )
   {
      throw paramError();
   }
}

FALCON_DEFINE_METHOD_P1( ClassRE, replaceAll )
{
   if( ! internal_change( ctx, 1 ) )
   {
      throw paramError();
   }
}

FALCON_DEFINE_METHOD_P1( ClassRE, change )
{
   if( !  internal_change( ctx, 2 ) )
   {
      throw paramError();
   }
}

FALCON_DEFINE_METHOD_P1( ClassRE, changeAll )
{
   if( ! internal_change( ctx, 3 ) )
   {
      throw paramError();
   }
}


FALCON_DEFINE_METHOD_P1( ClassRE, consume )
{
   Item* i_target = ctx->param(0);
   Item* i_getAll = ctx->param(1);

   if( i_target == 0|| ! i_target->isString() )
   {
      throw paramError();
   }

   bool getAll = i_getAll != 0 && i_getAll->isTrue();

   re2::RE2* re = static_cast<re2::RE2*>(ctx->self().asInst());
   String* target = i_target->asString();

   re2::StringPiece captured[MAX_CAPTURE_COUNT];
   int cc = re->NumberOfCapturingGroups() + 1;
   // paranoid...
   if( cc > MAX_CAPTURE_COUNT )
   {
      cc = MAX_CAPTURE_COUNT;
   }

   String* cfr = ctx->topData().asString();
   re2::StringPiece text(*cfr);

   bool match = re->Match(text,
              0,
              text.size(),
              re2::RE2::UNANCHORED,
              captured,
              cc);

   if( match )
   {
      int consumed = captured[0].end() - text.begin();
      text.remove_prefix(consumed);
      target->fromUTF8(text.ToString().c_str());

      if( cc  > 1 || getAll )
      {
         ItemArray* res = new ItemArray;
         res->reserve(cc);

         for( int i = getAll ? 0 : 1; i < cc; ++i )
         {
            std::string result = captured[i].ToString();
            String* scapt = new String( result.c_str(), result.size() );
            res->append( FALCON_GC_HANDLE( scapt ) );
         }
         ctx->returnFrame(FALCON_GC_HANDLE(res));
      }
      else {
         ctx->returnFrame(Item().setBoolean(true));
      }
   }
   else
   {
      ctx->returnFrame();
   }
}

FALCON_DEFINE_METHOD_P1( ClassRE, consumeMatch )
{
   Item* i_target = ctx->param(0);

   if( i_target == 0|| ! i_target->isString() )
   {
      throw paramError();
   }

   re2::RE2* re = static_cast<re2::RE2*>(ctx->self().asInst());
   String* target = i_target->asString();

   re2::StringPiece captured[36];
   int cc = re->NumberOfCapturingGroups() + 1;
   // paranoid...
   if( cc > 36 )
   {
      cc = 36;
   }

   String* cfr = ctx->topData().asString();
   re2::StringPiece text(*cfr);

   bool match = re->Match(text,
              0,
              text.size(),
              re2::RE2::UNANCHORED,
              captured,
              cc);

   if( match )
   {
      int consumed = captured[0].end() - text.begin();
      text.remove_prefix(consumed);
      target->fromUTF8(text.ToString().c_str());

      String* res = new String;
      res->fromUTF8(captured[0].ToString().c_str());
      ctx->returnFrame(FALCON_GC_HANDLE(res));
   }
   else
   {
      ctx->returnFrame();
   }
}

}

/* end of classre.cpp */
