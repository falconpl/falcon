/*
   FALCON - The Falcon Programming Language.
   FILE: exprproto.cpp

   Syntactic tree item definitions -- prototype generator.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 16:55:07 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprproto.cpp"

#include <falcon/vmcontext.h>
#include <falcon/prototypeclass.h>
#include <falcon/flexydict.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/stdhandlers.h>

#include <falcon/psteps/exprproto.h>
#include <falcon/ov_names.h>

#include <falcon/error.h>
#include <falcon/errors/codeerror.h>

#include <vector>

namespace Falcon
{

class ExprProto::Private
{
public:
   typedef std::vector< std::pair<String, Expression*> > DefVector;
   DefVector m_defs;

   Private() {}
   ~Private()
   {
      DefVector::iterator iter = m_defs.begin();
      while( m_defs.end() != iter )
      {
         dispose( iter->second );
         ++iter;
      }
   }
};


ExprProto::ExprProto( int line, int chr ):
   Expression( line, chr ),
   _p(new Private)
{
   FALCON_DECLARE_SYN_CLASS( expr_genproto )
   apply=apply_;
   m_trait = e_trait_composite;
}

ExprProto::ExprProto( const ExprProto& other ):
   Expression(other),
   _p(new Private)
{
   apply=apply_;
   m_trait = e_trait_composite;
}

ExprProto::~ExprProto()
{
   delete _p;
}


int32 ExprProto::arity() const
{
   return (int) _p->m_defs.size();
}

TreeStep* ExprProto::nth( int32 n ) const
{
   if( n < 0 ) n = (int)_p->m_defs.size() + n;
   if( n < 0 || n >= (int)_p->m_defs.size() ) return 0;
   return _p->m_defs[n].second;
}

bool ExprProto::setNth( int32 n, TreeStep* ts )
{
   if( ts->category() != TreeStep::e_cat_expression ) return false;
   if( n < 0 ) n = (int)_p->m_defs.size() + n;
   if( n < 0 || n >= (int)_p->m_defs.size() ) return false;
   if( ! ts->setParent(this) ) return false;
   dispose( _p->m_defs[n].second );
   _p->m_defs[n].second = static_cast<Expression*>(ts);
   return true;
}


bool ExprProto::remove( int n )
{
   if( n < 0 ) n = (int)_p->m_defs.size() + n;
   if( n < 0 || n >= (int)_p->m_defs.size() ) return false;
   dispose( _p->m_defs[n].second );
   _p->m_defs.erase( _p->m_defs.begin() + n );
   return true;
}


size_t ExprProto::size() const
{
   return _p->m_defs.size();
}


Expression* ExprProto::exprAt( size_t n ) const
{
   return _p->m_defs[n].second;
}


const String& ExprProto::nameAt( size_t n ) const
{
   return _p->m_defs[n].first;
}


ExprProto& ExprProto::add( const String& name, Expression* e )
{
   _p->m_defs.push_back( std::make_pair( name, e ) );
   return *this;
}


void ExprProto::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );
   tw->write( "p{" );
   if( !_p->m_defs.empty() )
   {
      Private::DefVector::const_iterator iter = _p->m_defs.begin();
      tw->write( "\n" );
      while( _p->m_defs.end() != iter )
      {

         tw->write( renderPrefix(depth+1) );
         tw->write(iter->first);
         tw->write( " = " );
         iter->second->render( tw, relativeDepth(depth) );
         tw->write( "\n" );
         ++iter;
      }

      tw->write( renderPrefix(depth) );
   }

   tw->write( "}" );

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }
}

bool ExprProto::simplify( Item& ) const
{
   // TODO, create a Proto value?
   return false;
}

void ExprProto::apply_( const PStep* ps, VMContext* ctx )
{
   static Class* cls =  Engine::handlers()->protoClass();

   const ExprProto* self = static_cast<const ExprProto*>(ps);
   CodeFrame& cs = ctx->currentCode();

   // apply all the expressions.
   int& seqId = cs.m_seqId;
   Private::DefVector& dv = self->_p->m_defs;
   int size = (int) dv.size();
   while( seqId < size )
   {
      Expression* current = dv[seqId++].second;
      if( ctx->stepInYield( current, cs ) )
      {
         return;
      }
   }

   // we're done with the exrpessions
   FlexyDict *value = new FlexyDict;
   Item retval = FALCON_GC_STORE( cls, value );

   Item* result = ctx->opcodeParams(size);
   Private::DefVector::iterator viter = dv.begin();
   while( viter != dv.end() )
   {
      if( viter->first == FALCON_PROTOTYPE_PROPERTY_OVERRIDE_BASE )
      {
         if( result->isArray() )
         {
            value->base().resize(0);
            value->base().merge( *result->asArray() );
         }
         else
         {
            value->base().resize(1);
            result->copied();
            value->base()[0] = *result;
         }
      }
      else if ( viter->first == FALCON_PROTOTYPE_PROPERTY_OVERRIDE_META )
      {
         if (! result->isUser() || result->asClass()->typeID() != FLC_CLASS_ID_PROTO )
         {
            throw FALCON_SIGN_ERROR( CodeError, e_meta_not_proto );
         }

         FlexyDict* meta = static_cast<FlexyDict*>(result->asInst());
         value->meta( meta );
      }
      else if ( viter->first == FALCON_PROTOTYPE_PROPERTY_OVERRIDE_ISBASE )
      {
         value->setBaseType(result->isTrue());
         value->insert(viter->first, *result );
      }
      else
      {
         value->insert(viter->first, *result );
      }

      ++result;
      ++viter;
   }

   // we're done.
   ctx->popCode();
   ctx->stackResult( size, retval );
}

}

/* end of exprproto.cpp */
