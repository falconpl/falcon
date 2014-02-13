#include "gtk_Spinner.hpp"

namespace Falcon {
namespace Gtk {

void Spinner::modInit( Module* mod )
{
    const Symbol* c_Spinner = mod->addClass( "GtkSpinner", &Spinner::init );
    Falcon::InheritDef *def = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkDrawingArea" ));

    c_Spinner->getClassDef()->addInheritance( def );

    c_Spinner->setWKS( true );
    c_Spinner->getClassDef()->factory( &Spinner::factory );

    MethodTab methods[] =
    {
    { "start",  &Spinner::start },
    { "stop",   &Spinner::stop },
    { NULL, NULL }
    };

    for( Gtk::MethodTab *currentMethod = methods; currentMethod->name; ++currentMethod )
    {
        mod->addClassMethod( c_Spinner, currentMethod->name, currentMethod->cb );
    }

}

Spinner::Spinner( const CoreClass* gen, const GtkSpinner* spinner ):
    CoreGObject( gen, (const GObject*) spinner )
{}


CoreObject* Spinner::factory( const CoreClass *gen, void *spinner, bool )
{
    return new Spinner( gen, (const GtkSpinner *)spinner );
}


FALCON_FUNC Spinner::init( VMARG )
{
    NO_ARGS;
    MYSELF;

    self->setObject( (void *) gtk_spinner_new() );
}


FALCON_FUNC Spinner::start( VMARG )
{
    NO_ARGS;
    MYSELF;
    GET_OBJ( self );

    gtk_spinner_start( (GtkSpinner *) _obj );
}


FALCON_FUNC Spinner::stop( VMARG )
{
    NO_ARGS;
    MYSELF;
    GET_OBJ( self );

    gtk_spinner_stop( (GtkSpinner *) _obj );
}

}
}
