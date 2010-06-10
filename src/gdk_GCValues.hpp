#ifndef GDK_GCVALUES_HPP
#define GDK_GCVALUES_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::GCValues
 */
class GCValues
    :
    public Falcon::CoreObject
{
public:

    GCValues( const Falcon::CoreClass*, const GdkGCValues* = 0 );

    ~GCValues();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkGCValues* getGCValues() const { return (GdkGCValues*) &m_gcvalues; }

    void setGCValues( const GdkGCValues* );

private:

    /*
     *  Increment ref count of internal objects.
     */
    void incref();

    /*
     *  Decrement ref count of internal objects.
     */
    void decref();

    GdkGCValues     m_gcvalues;

};


} // Gdk
} // Falcon

#endif // !GDK_GCVALUES_HPP
