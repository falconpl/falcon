/*
   FALCON - The Falcon Programming Language.
   FILE: stripoldata.cpp

   Pre-compiled data for string interpolation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 01 Feb 2013 11:06:19 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/stripoldata.cpp"

#include <falcon/setup.h>
#include <falcon/stripoldata.h>
#include <falcon/item.h>
#include <falcon/treestep.h>
#include <falcon/vmcontext.h>
#include <falcon/mt.h>
#include <falcon/dyncompiler.h>
#include <falcon/syntree.h>
#include <falcon/symbol.h>
#include <falcon/format.h>
#include <falcon/stdhandlers.h>

#include <vector>

namespace Falcon
{

class StrIPolData::Private
{
public:
   Mutex m_mtx;
   typedef std::vector<StrIPolData::Slice*> SliceList;
   SliceList m_slices;

   // true if the compiled slices are ours.
   bool m_ownCompiled;

   class PStepFormatter: public PStep
   {
   public:
      PStepFormatter() {apply = apply_;}
      virtual ~PStepFormatter() {}
      static void apply_( const PStep*, VMContext* ctx )
      {
         Item ivalue(ctx->opcodeParam(0));
         Item& iformat = ctx->opcodeParam(1);

         fassert( iformat.asClass() == Engine::handlers()->formatClass() );

         Format* fmt = static_cast<Format*>(iformat.asInst());
         ctx->popData(2);
         ctx->popCode();

         fmt->opFormat(ctx, ivalue );
      }
      virtual void describeTo( String& desc , int ) const
      {
         desc = "StrIPolData::PStepFormatter";
      }
   };
   PStepFormatter m_pStepFormatter;

   class PStepStringifier: public PStep
   {
   public:
      PStepStringifier() {apply = apply_;}
      virtual ~PStepStringifier() {}
      static void apply_( const PStep*, VMContext* ctx )
      {
         // for now, do nothing
         ctx->popCode();
         Class* cls;
         void* inst;
         ctx->topData().forceClassInst(cls, inst);
         cls->op_toString( ctx, inst );
      }
      virtual void describeTo( String& desc , int ) const
      {
         desc = "StrIPolData::PStepStringifier";
      }
   };
   PStepStringifier m_pStepStringifier;

   Private():
      m_ownCompiled(true)
      {}
   ~Private()
   {
      clear();
   }

   void clear() {

      if( m_ownCompiled )
      {
         SliceList::iterator iter = m_slices.begin();
         SliceList::iterator end = m_slices.end();
         while( iter != end )
         {
            Slice* sl = *iter;
            delete sl;
            ++iter;
         }
      }
      m_slices.clear();
   }

   void gcMark( uint32 mark ) {
      if( m_ownCompiled )
      {
         return;
      }

      SliceList::iterator iter = m_slices.begin();
      SliceList::iterator end = m_slices.end();
      while( iter != end )
      {
         Slice* sl = *iter;
         if( sl->m_compiled != 0 )
         {
            sl->m_compiled->gcMark(mark);
         }
         ++iter;
      }
      m_slices.clear();
   }

   void copy( const Private* other )
   {
      SliceList::const_iterator iter = other->m_slices.begin();
      SliceList::const_iterator end = other->m_slices.end();
      while( iter != end )
      {
         Slice* sl = *iter;
         Slice* mine = new Slice(*sl);
         m_slices.push_back(mine);
         ++iter;
      }

      // no need to put the stuff in the GC
      m_ownCompiled = true;
   }

   void setList( const SliceList& source )
   {
      clear();
      m_slices = source;
   }
};


void StrIPolData::PStepExprComp::apply_( const PStep* ps, VMContext* ctx )
{
   const PStepExprComp* self = static_cast<const PStepExprComp*>(ps);
   CodeFrame& cf = ctx->currentCode();

   TRACE( "StrIPolData::PStepExprComp::apply_ step %d", cf.m_seqId );

   TreeStep* ts = static_cast<TreeStep*>(ctx->topData().asInst());
   // we're out of business, but first add the expression to our owner.
   StrIPolData* owner = self->m_owner;
   Slice* slc = owner->getSlice(cf.m_seqId);
   slc->m_compiled = ts;

   // we're out of business
   ctx->resetCode(ts);
}

//=====================================================================================================
// Main class
//

StrIPolData::StrIPolData():
   GenericData("StrIPolData"),
   m_mark(0),
   m_dynCount(0),
   m_pStepExprComp(this)
{
   _p = new Private;
}

StrIPolData::StrIPolData(const StrIPolData& other):
   GenericData("StrIPolData"),
   m_mark(0),
   m_dynCount( other.m_dynCount ),
   m_pStepExprComp(this)
{
   _p = new Private;
   _p->copy(other._p);
}


StrIPolData::~StrIPolData()
{
   delete _p;
}


StrIPolData::t_parse_result StrIPolData::parse( const String& source, int& failPos )
{
   typedef enum {
      e_basic,
      e_dollar,
      e_paropen,
      e_symbol,
      e_expr,
      e_formatspec,
      e_colonspec
   }
   t_state;

   t_state state = e_basic;
   uint32 curpos = 0;
   uint32 size = source.length();

   uint32 parCount = 0;
   uint32 graphCount = 0;

   Private::SliceList list;
   bool fail = false;

   String text;
   String expr;
   String format;

   int32 dynCount = 0;

   while( curpos < size )
   {
      uint32 chr = source.getCharAt(curpos);

      switch( state )
      {
      case e_basic:
         if( chr == '$' )
         {
            state = e_dollar;
         }
         else {
            text.append(chr);
         }
         break;

      case e_dollar:
        if( chr == '$' )
        {
           state = e_basic;
           text.append('$');
        }
        else
        {
           if( chr == '(' )
           {
              parCount = 1;
              state = e_paropen;
              expr.size(0);
           }
           else if( chr == '{' )
           {
              graphCount = 1;
              state = e_expr;
              expr.size(0);
           }
           else if( !( (chr >= 'A' && chr <='Z' ) || (chr >= 'a' && chr <='z' ) || chr > 127 ))
           {
              fail = true;
              break;
           }
           else {
              state = e_symbol;
              expr.size(0);
              expr.append(chr);
           }

           // curpos is on '$', we didn't advance it yet.
           if( text.size() != 0 )
           {
              list.push_back( new Slice(Slice::e_t_static, text) );
              text.size(0);
           }
        }
        break;

      case e_paropen:
         if( chr == '{' )
         {
            graphCount++;
            state = e_expr;
            expr.size(0);
         }
         else if ( (chr >= 'A' && chr <='Z' ) || (chr >= 'a' && chr <='z' ) || chr > 127 )
         {
            state = e_symbol;
            expr.size(0);
            expr.append(chr);
         }
         else
         {
            fail = true;
         }
         break;


      case e_symbol:
         if( chr == ')' && parCount )
         {
            // we're done -- can't be long 0 or we'd have cought it
            list.push_back( new Slice( expr == "self" ? Slice::e_t_expr : Slice::e_t_symbol, expr) );
            dynCount++;
            state = e_basic;
            parCount = 0;
         }
         else if( chr == ':' && parCount )
         {
            list.push_back( new Slice( expr == "self" ? Slice::e_t_expr : Slice::e_t_symbol, expr) );
            dynCount++;
            state = e_formatspec;
            format.size(0);
         }
         else if (
                  (chr>='0' && chr <= '9')
                  || (chr >= 'A' && chr <='Z' )
                  || (chr >= 'a' && chr <='z' )
                  || chr > 127 )
         {
            expr.append(chr);
         }
         else
         {
            list.push_back( new Slice(expr == "self" ? Slice::e_t_expr : Slice::e_t_symbol, expr) );
            dynCount++;
            state = e_basic;
            text.size(0);
            text.append(chr);
         }
         break;

      case e_expr:
         if( chr == '{' )
         {
            graphCount++;
            expr.append(chr);
         }
         else if( chr == '}' )
         {
            graphCount--;
            if( graphCount == 0 )
            {
               if( parCount == 0 )
               {
                  // we're done.
                  if( expr.size() == 0 )
                  {
                     fail = true;
                  }
                  else {
                     list.push_back( new Slice(Slice::e_t_expr, expr) );
                     dynCount++;
                     state = e_basic;
                  }
               }
               else
               {
                  list.push_back( new Slice(Slice::e_t_expr, expr) );
                  dynCount++;
                  state = e_colonspec;
                  format.size(0);
               }
            }
            else {
               expr.append(chr);
            }
         }
         else {
            expr.append(chr);
         }
         break;

      case e_formatspec:
         if( chr == ' ' )
         {
            // do nothing
         }
         else if( chr == ')' )
         {
            if( format.size() == 0 )
            {
               fail = true;
            }
            else {
               list.back()->m_format = new Format;
               list.back()->m_format->parse(format);
               state = e_basic;
               parCount = 0;
            }
         }
         else {
            format.append(chr);
         }
         break;

      case e_colonspec:
         if( chr == ' ' )
         {
            // do nothing
         }
         else if( chr == ')' )
         {
            // we're done, without format.
            state = e_basic;
         }
         else if( chr == ':' )
         {
            state = e_formatspec;
         }
         else {
            format.append(chr);
            state = e_formatspec;
         }
         break;

      } // switch state


      if( fail )
      {
         failPos = curpos;
         for( uint32 i = 0; i < list.size();++i) {
            delete list[i];
         }
         return e_pr_fail;
      }
      curpos++;
   }

   if( state == e_symbol && parCount == 0 )
   {
      // put in the last symbol.
      list.push_back( new Slice(expr == "self" ? Slice::e_t_expr : Slice::e_t_symbol, expr) );
      dynCount++;
      state = e_basic;
   }

   if( state != e_basic )
   {
      failPos = curpos;
      for( uint32 i = 0; i < list.size();++i) {
         delete list[i];
      }
      return e_pr_fail;
   }

   // if we din't find any dynslice, ignore the process.
   if( list.empty() )
   {
      return e_pr_noneed;
   }

   // eventually push in the last entity
   if( text.size() != 0 )
   {
      list.push_back( new Slice(Slice::e_t_static, text) );
   }

   _p->m_mtx.lock();
   //m_source = source;
   _p->setList(list);
   m_dynCount = dynCount;
   _p->m_mtx.unlock();

   return e_pr_ok;
}


void StrIPolData::gcMark( uint32 value )
{
   if( m_mark != value )
   {
      m_mark = value;
      _p->gcMark(value);
   }
}


bool StrIPolData::gcCheck( uint32 value )
{
   return m_mark >= value;
}

StrIPolData* StrIPolData::clone() const
{
   return new StrIPolData(*this);
}


void StrIPolData::describe( String& target ) const
{
   if( _p->m_slices.empty())
   {
      target = "/* Empty string interpolation */";
   }
   else {
      target = "/* data for */ @\"";

      // test the stuff:
      for( uint32 i = 0; i < _p->m_slices.size();++i)
      {
         Slice* sl = _p->m_slices[i];
         switch( sl->m_type )
         {
         case Slice::e_t_static:
            {
               String temp = sl->m_def;
               for( uint32 pos = temp.find('$'); pos != String::npos; pos = temp.find('$', pos+2))
               {
                  temp.insert(pos, 0, "$");
               }
               target += temp;
            }
            break;

         case Slice::e_t_symbol:
            target += '$';
            if( sl->m_format == 0 )
            {
               target += "(" + sl->m_def + ")";
            }
            else {
               target += "(" + sl->m_def + ":" + sl->m_format->originalFormat() + ")";
            }
            break;

         case Slice::e_t_expr:
            target += '$';
            if( sl->m_format == 0 )
            {
               target += "{" +  sl->m_def + "}";
            }
            else {
               target += "({" + sl->m_def + "}:" + sl->m_format->originalFormat() + ")";
            }
            break;
         }
      }
   }

   target += '"';
}

uint32 StrIPolData::addSlice( Slice* slice )
{
   _p->m_mtx.lock();
   _p->m_slices.push_back(slice);
   uint32 size = _p->m_slices.size();
   if( slice->m_type != Slice::e_t_static )
   {
      m_dynCount++;
   }
   _p->m_mtx.unlock();

   return size;
}

void StrIPolData::delSlice( uint32 pos )
{
   _p->m_mtx.lock();
   if( pos > _p->m_slices.size() )
   {
      _p->m_mtx.unlock();
      return;
   }

   Slice* sl = _p->m_slices[pos];
   _p->m_slices.erase(_p->m_slices.begin()+pos);
   if( sl->m_type != Slice::e_t_static )
   {
      m_dynCount--;
   }
   _p->m_mtx.unlock();

   delete sl;
}

StrIPolData::Slice* StrIPolData::getSlice( uint32 pos ) const
{
   _p->m_mtx.lock();
   if( pos > _p->m_slices.size() )
   {
      _p->m_mtx.unlock();
      return 0;
   }

   Slice* sl = _p->m_slices[pos];
   _p->m_mtx.unlock();

   return sl;
}

uint32 StrIPolData::sliceCount() const
{
   _p->m_mtx.lock();
   uint32 size = _p->m_slices.size();
   _p->m_mtx.unlock();

   return size;
}

String* StrIPolData::mount( Item* data ) const
{
   String* target = new String;
   uint32 dyns = 0;
   _p->m_mtx.lock();
   uint32 size = _p->m_slices.size();
   for( uint32 i = 0 ; i < size; ++i )
   {
      Slice* slice = _p->m_slices[i];
      if( slice->m_type == Slice::e_t_static )
      {
         target->append( slice->m_def );
      }
      else
      {
         fassert( dyns < m_dynCount );
         const Item& item = data[dyns];
         if( ! item.isString() ) {
            _p->m_mtx.unlock();

            delete target;
            return 0;
         }

         String* str = item.asString();
         target->append(*str);

         ++dyns;
      }
   }
   _p->m_mtx.unlock();

   return target;
}

bool StrIPolData::prepareStep( VMContext* ctx , uint32 id )
{
   _p->m_mtx.lock();
   if( id >= _p->m_slices.size() )
   {
      _p->m_mtx.unlock();
      return false;
   }
   Slice* slice = _p->m_slices[id];
   _p->m_mtx.unlock();

   if( slice->m_type == Slice::e_t_static )
   {
      return false;
   }

   if( slice->m_type == Slice::e_t_symbol )
   {
      if( slice->m_symbol == 0 )
      {
         slice->m_symbol = Engine::getSymbol(slice->m_def);
      }
      Item* value = ctx->resolveSymbol(slice->m_symbol, false);
      fassert( value != 0 );

      if( slice->m_format != 0 )
      {
         ctx->pushCode(&_p->m_pStepFormatter);
         // the format string is in us, and we're alive as we're below in the data stack,
         // there's no need to garbage the string.
         ctx->pushData( Item(slice->m_format->handler(), slice->m_format) );
         ctx->pushData(*value);
      }
      else
      {
         ctx->pushData(*value);

         if( ! value->isString() )
         {
            Class* cls;
            void* data;
            value->forceClassInst(cls, data);
            cls->op_toString(ctx, data);
         }
      }
   }
   else {
      // it's an expression.
      if( slice->m_format != 0 )
      {
         ctx->pushCode(&_p->m_pStepFormatter);
         // the format string is in us, and we're alive as we're below in the data stack,
         // there's no need to garbage the string.
         ctx->pushData( Item(slice->m_format->handler(), slice->m_format) );
      }
      else {
         ctx->pushCode(&_p->m_pStepStringifier);
      }

      // do we have the expression?
      if( slice->m_compiled == 0 )
      {
         DynCompiler dyncomp(ctx);
         // compile it (and possibly save it).
         ctx->pushCode(&m_pStepExprComp);

         CodeFrame& cf = ctx->currentCode();
         cf.m_seqId = id;
         SynTree* tree = dyncomp.compile( slice->m_def );
         if( &cf != &ctx->currentCode() )
         {
            // let the step handle the situation.
            ctx->pushData(FALCON_GC_HANDLE(tree));
            return true;
         }
         ctx->popCode();

         if( tree->size() == 1 )
         {
            slice->m_compiled = tree->detach(0);
            delete tree;
         }
         else
         {
            slice->m_compiled = tree;
         }

         // set the line where this thing is compiled
         slice->m_compiled->decl(line(),1);
      }

      ctx->pushCode( slice->m_compiled );
   }

   return true;
}

//==============================================================================
//
StrIPolData::Slice::Slice( t_type t, const String& def, Format* format, TreeStep* comp ):
   m_type(t),
   m_def(def),
   m_format(format),
   m_compiled(comp),
   m_symbol(0)
{

}

StrIPolData::Slice::Slice( const Slice& other ):
   m_type(other.m_type),
   m_def(other.m_def),
   m_format(0),
   m_compiled(0),
   m_symbol(0)
{
   if( other.m_compiled != 0 )
   {
      m_compiled = other.m_compiled->clone();
   }

   if( other.m_format != 0 )
   {
      m_format = other.m_format->clone();
   }
}

StrIPolData::Slice::~Slice()
{
   delete m_compiled;
   delete m_format;
   if( m_symbol != 0 )
   {
      m_symbol->decref();
   }
}

}

/* end of stripoldata.cpp */
