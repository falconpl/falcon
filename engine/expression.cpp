/*
   FALCON - The Falcon Programming Language.
   FILE: expression.cpp

   Syntactic tree item definitions -- expression elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Bgin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/expression.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/pcode.h>
#include <falcon/trace.h>

namespace Falcon {

Expression::Expression( const Expression &other ):
   PStep(other),
   m_pstep_lvalue(0),
   m_operator( other.m_operator ),
   m_sourceRef( other.m_sourceRef )
{}

Expression::~Expression()
{}

void Expression::serialize( DataWriter* s ) const
{
   byte type = (byte) m_operator;
   s->write( type );
   m_sourceRef.serialize( s );
}

void Expression::deserialize( DataReader* s )
{
   m_sourceRef.deserialize( s );
}

void Expression::precompile( PCode* pcode ) const
{
   pcode->pushStep( this );
}

void Expression::precompileLvalue( PCode* pcode ) const
{
   // We do this, but the parser should have blocked us...
   pcode->pushStep( this );
}

void Expression::precompileAutoLvalue( PCode* pcode, const PStep* activity, bool, bool ) const
{
   // We do this, but the parser should have blocked us...
   precompile( pcode );             // access -- prepare params
   // no save
   pcode->pushStep( activity );     // action
   // no restore
   precompileLvalue( pcode );       // storage -- if applicable.
}

//=============================================================

UnaryExpression::UnaryExpression( const UnaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() )
{
}

UnaryExpression::~UnaryExpression()
{
   delete m_first;
}


void UnaryExpression::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling un-exp: %p (%s)", pcode, describe().c_ize() );
   m_first->precompile( pcode );
   pcode->pushStep( this );
}

void UnaryExpression::serialize( DataWriter* s ) const
{
   Expression::serialize( s );
   m_first->serialize( s );
}

void UnaryExpression::deserialize( DataReader* s )
{
   Expression::deserialize(s);
}

bool UnaryExpression::isStatic() const
{
   return m_first->isStatic();
}

//=============================================================

BinaryExpression::BinaryExpression( const BinaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() ),
   m_second( other.m_second->clone() )
{
}

BinaryExpression::~BinaryExpression()
{
   delete m_first;
   delete m_second;
}


void BinaryExpression::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling bin-exp: %p (%s)", pcode, describe().c_ize() );
   m_first->precompile( pcode );
   m_second->precompile( pcode );
   pcode->pushStep( this );
}

void BinaryExpression::serialize( DataWriter* s ) const
{
   Expression::serialize( s );
   m_first->serialize( s );
   m_second->serialize( s );
}

void BinaryExpression::deserialize( DataReader* s )
{
   Expression::deserialize(s);
}

bool BinaryExpression::isStatic() const
{
   return m_first->isStatic() && m_second->isStatic();
}

//=============================================================
TernaryExpression::TernaryExpression( const TernaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() ),
   m_second( other.m_second->clone() ),
   m_third( other.m_third->clone() )
{
}

TernaryExpression::~TernaryExpression()
{
   delete m_first;
   delete m_second;
   delete m_third;
}

void TernaryExpression::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling tri-exp: %p (%s)", pcode, describe().c_ize() );
   
   m_first->precompile( pcode );
   m_second->precompile( pcode );
   m_third->precompile( pcode );
   pcode->pushStep( this );
}


void TernaryExpression::serialize( DataWriter* s ) const
{
   Expression::serialize( s );
   m_third->serialize( s );
   m_second->serialize( s );
   m_first->serialize( s );
}

void TernaryExpression::deserialize( DataReader* s )
{
   Expression::deserialize(s);
}

bool TernaryExpression::isStatic() const
{
   return m_first->isStatic() && m_second->isStatic() && m_third->isStatic();
}

}

/* end of expression.cpp */
