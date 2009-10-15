/*
   FALCON - The Falcon Programming Language.
   FILE: complex.h

   Complex class for Falcon
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 09 Sep 2009 23:09:25 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Complex class for Falcon
   Internal logic functions - declarations.
*/

#ifndef FALCON_complex_H
#define FALCON_complex_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/basealloc.h>

namespace Falcon {


class Complex: public BaseAlloc
{
   numeric m_real;
   numeric m_imag;

   void throw_div_by_zero();

public:

   Complex( numeric r=0, numeric i=0 ):
      m_real(r),
      m_imag(i)
      {}

   Complex( const Complex& other ):
      m_real( other.m_real ),
      m_imag( other.m_imag )
      {}

	~Complex ( ) {}

   inline numeric real() const { return m_real; }
   inline numeric imag() const { return m_imag; }
   inline void real( numeric r ) { m_real = r; }
   inline void imag( numeric i ) { m_imag = i; }

   //=============================================
   // Math operators
   //

   inline Complex operator +( const Complex &other )
   {
      return Complex( m_real + other.m_real, m_imag + other.m_imag );
   }

   inline Complex operator -( const Complex &other )
   {
      return Complex( m_real - other.m_real, m_imag - other.m_imag );
   }

   // (ac−bd,bc+ad)
   inline Complex operator *( const Complex &other )
   {
      return Complex( m_real * other.m_real - m_imag * other.m_imag,
                      m_imag * other.m_real + m_real * other.m_imag );
   }

	//(ac+bd+i(bc-ad))/(c2+d2)
   inline Complex operator /( const Complex &other )
   {
      numeric divisor =  other.m_real*other.m_real + other.m_imag * other.m_imag;
      if ( divisor == 0 )
         throw_div_by_zero(); // don't want this inline.

      return Complex(
               (m_real * other.m_real + m_imag * other.m_imag) / divisor,
               (m_imag * other.m_real - m_real * other.m_imag) / divisor );
   }

   numeric abs() const;
   Complex conj() const;
   
   //=============================================
   // Assignment operators
   //

   inline Complex &operator =( const Complex &other )
   {
      m_real = other.m_real;
      m_imag = other.m_imag;
      return *this;
   }

   inline Complex& operator +=( const Complex &other )
   {
      m_real += other.m_real;
      m_imag += other.m_imag;
      return *this;     
   }

   inline Complex& operator -=( const Complex &other )
   {
      m_real -= other.m_real;
      m_imag -= other.m_imag;
      return *this;
   }

   inline Complex& operator *=( const Complex &other )
   {
      m_real = m_real * other.m_real - m_imag * other.m_imag;
      m_imag = m_imag * other.m_real + m_real * other.m_imag;
      return *this;
   }

   inline Complex& operator /=( const Complex &other )
   {
      *this = *this / other;
      return *this;
   }

   //=============================================
   // Relational operators
   //

   inline bool operator ==( const Complex &other )
   {
      return m_real == other.m_real && m_imag == other.m_imag;
   }

   inline bool operator !=( const Complex &other )
   {
      return m_real != other.m_real || m_imag != other.m_imag;
   }

   inline bool operator >( const Complex &other )
   {
      return m_real > other.m_real || (m_real == other.m_real && m_imag > other.m_imag);
   }

   inline bool operator >=( const Complex &other )
   {
      return m_real >= other.m_real || (m_real == other.m_real && m_imag >= other.m_imag);
   }

   inline bool operator <( const Complex &other )
   {
      return m_real < other.m_real || (m_real == other.m_real && m_imag < other.m_imag);
   }

   inline bool operator <=( const Complex &other )
   {
      return m_real < other.m_real || (m_real == other.m_real && m_imag < other.m_imag);
   }

};

}

#endif

/* end of complex.h */
