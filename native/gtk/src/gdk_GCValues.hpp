#ifndef GDK_GCVALUES_HPP
#define GDK_GCVALUES_HPP

#include "modgtk.hpp"

#define GET_GCVALUES( item ) \
        (((Gdk::GCValues*) (item).asObjectSafe())->getObject())

namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::GCValues
 */
class GCValues
    :
    public Gtk::VoidObject
{
public:

    GCValues( const Falcon::CoreClass*, const GdkGCValues* = 0 );

    GCValues( const GCValues& );

    ~GCValues();

    GCValues* clone() const { return new GCValues( *this ); }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkGCValues* getObject() const { return (GdkGCValues*) m_obj; }

    void setObject( const void* );

private:

    void incref();

    void decref();

};


} // Gdk
} // Falcon

#endif // !GDK_GCVALUES_HPP
