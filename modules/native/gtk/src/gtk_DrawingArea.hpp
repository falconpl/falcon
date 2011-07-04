#ifndef GTK_DRAWINGAREA_H
#define GTK_DRAWINGAREA_H

#include "modgtk.hpp"

namespace Falcon {
namespace Gtk {

class DrawingArea:
        public Gtk::CoreGObject
{
public:
    DrawingArea( const CoreClass*, const GtkDrawingArea* );

    static CoreObject* factory( const CoreClass*, void *, bool );

    static void modInit( Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC size( VMARG );
};

}
}

#endif // GTK_DRAWINGAREA_H
