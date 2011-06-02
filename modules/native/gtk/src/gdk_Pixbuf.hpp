#ifndef GDK_PIXBUF_HPP
#define GDK_PIXBUF_HPP

#include "modgtk.hpp"

#include <gdk-pixbuf/gdk-pixbuf.h>

#define GET_PIXBUF( item ) \
        (((Gdk::Pixbuf*) (item).asObjectSafe())->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Pixbuf
 */
class Pixbuf
    :
    public Gtk::CoreGObject
{
public:

    Pixbuf( const Falcon::CoreClass*, const GdkPixbuf* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkPixbuf* getObject() const { return (GdkPixbuf*) m_obj; }

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC version( VMARG );

    static FALCON_FUNC get_n_channels( VMARG );

    static FALCON_FUNC get_has_alpha( VMARG );

    static FALCON_FUNC get_bits_per_sample( VMARG );

    static FALCON_FUNC get_pixels( VMARG );

    static FALCON_FUNC get_width( VMARG );

    static FALCON_FUNC get_height( VMARG );

    static FALCON_FUNC new_from_file( VMARG );

    static FALCON_FUNC new_from_file_at_size( VMARG );

    static FALCON_FUNC new_from_file_at_scale( VMARG );

    static FALCON_FUNC flip( VMARG );

    static FALCON_FUNC rotate_simple( VMARG );

	static FALCON_FUNC scale_simple( VMARG );
  
};


} // Gdk
} // Falcon

#endif // !GDK_PIXBUF_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
