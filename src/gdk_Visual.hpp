#ifndef GDK_VISUAL_HPP
#define GDK_VISUAL_HPP

#include "modgtk.hpp"

#define GET_VISUAL( item ) \
        (((Gdk::Visual*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Visual
 */
class Visual
    :
    public Gtk::VoidObject
{
public:

    Visual( const Falcon::CoreClass*, const GdkVisual* = 0 );

    Visual( const Visual& );

    ~Visual();

    Visual* clone() const { return new Visual( *this ); }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkVisual* getObject() const { return (GdkVisual*) m_obj; }

    static FALCON_FUNC query_depths( VMARG );

    static FALCON_FUNC query_visual_types( VMARG );

    static FALCON_FUNC list_visuals( VMARG );

    static FALCON_FUNC get_best_depth( VMARG );

    static FALCON_FUNC get_best_type( VMARG );

    static FALCON_FUNC get_system( VMARG );

    static FALCON_FUNC get_best( VMARG );

    static FALCON_FUNC get_best_with_depth( VMARG );

    static FALCON_FUNC get_best_with_type( VMARG );

    static FALCON_FUNC get_best_with_both( VMARG );
#if 0
    static FALCON_FUNC ref( VMARG );

    static FALCON_FUNC unref( VMARG );
#endif
    static FALCON_FUNC get_screen( VMARG );

private:

    void incref() const;

    void decref() const;

};


} // Gdk
} // Falcon

#endif // !GDK_VISUAL_HPP
