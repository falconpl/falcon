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

#include <falcon/exprproto.h>
#include <falcon/vmcontext.h>
#include <falcon/pcode.h>
#include <falcon/prototypeclass.h>
#include <falcon/flexydict.h>

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


ExprProto::ExprProto():
   Expression( Expression::t_prototype ),
   _p(new Private)
{
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


void ExprProto::serialize( DataWriter* ) const
{
   // TODO
}

void ExprProto::deserialize( DataReader* )
{
   // TODO
}

void ExprProto::precompile( PCode* pcd ) const
{
   // push all the expressions in inverse order.
   Private::DefVector::const_iterator iter = _p->m_defs.begin();
   while( _p->m_defs.end() != iter )
   {
      iter->second->precompile( pcd );
      ++iter;
   }
   pcd->pushStep(this);

}

void ExprProto::describe( String& tgt ) const
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
      iter->second->oneLiner(temp);
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
   
   const ExprProto* expr = static_cast<const ExprProto*>(ps);
   Private::DefVector& dv = expr->_p->m_defs;
   register int size = (int) dv.size();

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
  
   ctx->stackResult( size, FALCON_GC_STORE( coll, cls, value ) );
}

}

/* end of exprproto.cpp */
