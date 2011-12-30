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

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <falcon/psteps/exprproto.h>

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
         delete iter->second;
         ++iter;
      }
   }
};


ExprProto::ExprProto( int line, int chr ):
   Expression( line, chr ),
   _p(new Private)
{
   FALCON_DECLARE_SYN_CLASS( expr_proto )
   apply=apply_;
}

ExprProto::ExprProto( const ExprProto& other ):
   Expression(other),
   _p(new Private)
{
   apply=apply_;
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
   return _p->m_defs[n]->second;
}

bool ExprProto::nth( int32 n, TreeStep* ts )
{
   if( ts->category() != TreeStep::e_cat_expression ) return false;
   if( n < 0 ) n = (int)_p->m_defs.size() + n;
   if( n < 0 || n >= (int)_p->m_defs.size() ) return false;
   if( ! ts->setParent(this) ) return false;
   delete _p->m_defs[n]->second;
   _p->m_defs[n]->second = ts;
   return true;
}


virtual bool ExprProto::remove( int n )
{
   if( n < 0 ) n = (int)_p->m_defs.size() + n;
   if( n < 0 || n >= (int)_p->m_defs.size() ) return false;
   delete _p->m_defs[n]->second;
   _p->m_defs.erase( _p->m_defs.begin() + n );
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


void ExprProto::describeTo( String& tgt, int depth ) const
{
   tgt.size(0);
   tgt += "p{";
   Private::DefVector::const_iterator iter = _p->m_defs.begin();
   String temp;
   while( _p->m_defs.end() != iter )
   {
      if ( tgt.size()>2 ) {
         tgt += ";";
      }

      temp.size(0);
      iter->second->describeTo(temp, depth+1);
      tgt += iter->first + "=" + temp;
      ++iter;
   }
}

bool ExprProto::simplify( Item& ) const
{
   // TODO, create a Proto value?
   return false;
}



void ExprProto::apply_( const PStep* ps, VMContext* ctx )
{
   static Collector* coll = Engine::instance()->collector();
   static Class* cls =  Engine::instance()->protoClass();
   
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

   Item* result = ctx->opcodeParams(size);
   Private::DefVector::iterator viter = dv.begin();
   while( viter != dv.end() )
   {
      // pre-methodize
      if( result->isFunction() )
      {
         Function* f = result->asFunction();
         result->setUser( cls, value, true );
         result->methodize( f );
      }

      if( viter->first == "_base" )
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
      else
      {
         value->insert(viter->first, *result );
      }
      
      ++result;
      ++viter;
   }
  
   // we're done.
   ctx->popCode();
   ctx->stackResult( size, FALCON_GC_STORE( coll, cls, value ) );
}

}

/* end of exprproto.cpp */
