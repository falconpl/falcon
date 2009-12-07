
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.4.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse         flc_src_parse
#define yylex           flc_src_lex
#define yyerror         flc_src_error
#define yylval          flc_src_lval
#define yychar          flc_src_char
#define yydebug         flc_src_debug
#define yynerrs         flc_src_nerrs


/* Copy the first part of user declarations.  */

/* Line 189 of yacc.c  */
#line 17 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


#include <math.h>
#include <stdio.h>
#include <iostream>

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/src_lexer.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <stdlib.h>

#include <falcon/fassert.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::Compiler *>(yyparam) )
#define  CTX_LINE  ( COMPILER->lexer()->contextStart() )
#define  LINE      ( COMPILER->lexer()->previousLine() )
#define  CURRENT_LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM yyparam
#define YYLEX_PARAM yyparam

int flc_src_parse( void *param );
void flc_src_error (const char *s);

inline int flc_src_lex (void *lvalp, void *yyparam)
{
   return COMPILER->lexer()->doLex( lvalp );
}

/* Cures a bug in bison 1.8  */
#undef __GNUC_MINOR__



/* Line 189 of yacc.c  */
#line 124 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     EOL = 258,
     INTNUM = 259,
     DBLNUM = 260,
     SYMBOL = 261,
     STRING = 262,
     NIL = 263,
     UNB = 264,
     END = 265,
     DEF = 266,
     WHILE = 267,
     BREAK = 268,
     CONTINUE = 269,
     DROPPING = 270,
     IF = 271,
     ELSE = 272,
     ELIF = 273,
     FOR = 274,
     FORFIRST = 275,
     FORLAST = 276,
     FORMIDDLE = 277,
     SWITCH = 278,
     CASE = 279,
     DEFAULT = 280,
     SELECT = 281,
     SELF = 282,
     FSELF = 283,
     TRY = 284,
     CATCH = 285,
     RAISE = 286,
     CLASS = 287,
     FROM = 288,
     OBJECT = 289,
     RETURN = 290,
     GLOBAL = 291,
     INIT = 292,
     LOAD = 293,
     LAUNCH = 294,
     CONST_KW = 295,
     EXPORT = 296,
     IMPORT = 297,
     DIRECTIVE = 298,
     COLON = 299,
     FUNCDECL = 300,
     STATIC = 301,
     INNERFUNC = 302,
     FORDOT = 303,
     LISTPAR = 304,
     LOOP = 305,
     ENUM = 306,
     TRUE_TOKEN = 307,
     FALSE_TOKEN = 308,
     STATE = 309,
     OUTER_STRING = 310,
     CLOSEPAR = 311,
     OPENPAR = 312,
     CLOSESQUARE = 313,
     OPENSQUARE = 314,
     DOT = 315,
     OPEN_GRAPH = 316,
     CLOSE_GRAPH = 317,
     ARROW = 318,
     VBAR = 319,
     ASSIGN_POW = 320,
     ASSIGN_SHL = 321,
     ASSIGN_SHR = 322,
     ASSIGN_BXOR = 323,
     ASSIGN_BOR = 324,
     ASSIGN_BAND = 325,
     ASSIGN_MOD = 326,
     ASSIGN_DIV = 327,
     ASSIGN_MUL = 328,
     ASSIGN_SUB = 329,
     ASSIGN_ADD = 330,
     OP_EQ = 331,
     OP_AS = 332,
     OP_TO = 333,
     COMMA = 334,
     QUESTION = 335,
     OR = 336,
     AND = 337,
     NOT = 338,
     LE = 339,
     GE = 340,
     LT = 341,
     GT = 342,
     NEQ = 343,
     EEQ = 344,
     PROVIDES = 345,
     OP_NOTIN = 346,
     OP_IN = 347,
     DIESIS = 348,
     ATSIGN = 349,
     CAP_CAP = 350,
     VBAR_VBAR = 351,
     AMPER_AMPER = 352,
     MINUS = 353,
     PLUS = 354,
     PERCENT = 355,
     SLASH = 356,
     STAR = 357,
     POW = 358,
     SHR = 359,
     SHL = 360,
     CAP_XOROOB = 361,
     CAP_ISOOB = 362,
     CAP_DEOOB = 363,
     CAP_OOB = 364,
     CAP_EVAL = 365,
     TILDE = 366,
     NEG = 367,
     AMPER = 368,
     DECREMENT = 369,
     INCREMENT = 370,
     DOLLAR = 371
   };
#endif
/* Tokens.  */
#define EOL 258
#define INTNUM 259
#define DBLNUM 260
#define SYMBOL 261
#define STRING 262
#define NIL 263
#define UNB 264
#define END 265
#define DEF 266
#define WHILE 267
#define BREAK 268
#define CONTINUE 269
#define DROPPING 270
#define IF 271
#define ELSE 272
#define ELIF 273
#define FOR 274
#define FORFIRST 275
#define FORLAST 276
#define FORMIDDLE 277
#define SWITCH 278
#define CASE 279
#define DEFAULT 280
#define SELECT 281
#define SELF 282
#define FSELF 283
#define TRY 284
#define CATCH 285
#define RAISE 286
#define CLASS 287
#define FROM 288
#define OBJECT 289
#define RETURN 290
#define GLOBAL 291
#define INIT 292
#define LOAD 293
#define LAUNCH 294
#define CONST_KW 295
#define EXPORT 296
#define IMPORT 297
#define DIRECTIVE 298
#define COLON 299
#define FUNCDECL 300
#define STATIC 301
#define INNERFUNC 302
#define FORDOT 303
#define LISTPAR 304
#define LOOP 305
#define ENUM 306
#define TRUE_TOKEN 307
#define FALSE_TOKEN 308
#define STATE 309
#define OUTER_STRING 310
#define CLOSEPAR 311
#define OPENPAR 312
#define CLOSESQUARE 313
#define OPENSQUARE 314
#define DOT 315
#define OPEN_GRAPH 316
#define CLOSE_GRAPH 317
#define ARROW 318
#define VBAR 319
#define ASSIGN_POW 320
#define ASSIGN_SHL 321
#define ASSIGN_SHR 322
#define ASSIGN_BXOR 323
#define ASSIGN_BOR 324
#define ASSIGN_BAND 325
#define ASSIGN_MOD 326
#define ASSIGN_DIV 327
#define ASSIGN_MUL 328
#define ASSIGN_SUB 329
#define ASSIGN_ADD 330
#define OP_EQ 331
#define OP_AS 332
#define OP_TO 333
#define COMMA 334
#define QUESTION 335
#define OR 336
#define AND 337
#define NOT 338
#define LE 339
#define GE 340
#define LT 341
#define GT 342
#define NEQ 343
#define EEQ 344
#define PROVIDES 345
#define OP_NOTIN 346
#define OP_IN 347
#define DIESIS 348
#define ATSIGN 349
#define CAP_CAP 350
#define VBAR_VBAR 351
#define AMPER_AMPER 352
#define MINUS 353
#define PLUS 354
#define PERCENT 355
#define SLASH 356
#define STAR 357
#define POW 358
#define SHR 359
#define SHL 360
#define CAP_XOROOB 361
#define CAP_ISOOB 362
#define CAP_DEOOB 363
#define CAP_OOB 364
#define CAP_EVAL 365
#define TILDE 366
#define NEG 367
#define AMPER 368
#define DECREMENT 369
#define INCREMENT 370
#define DOLLAR 371




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union 
/* Line 214 of yacc.c  */
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
lex_value_t
{

/* Line 214 of yacc.c  */
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"

   Falcon::int64 integer;
   Falcon::numeric numeric;
   char * charp;
   Falcon::String *stringp;
   Falcon::Symbol *symbol;
   Falcon::Value *fal_val;
   Falcon::Expression *fal_exp;
   Falcon::Statement *fal_stat;
   Falcon::ArrayDecl *fal_adecl;
   Falcon::DictDecl *fal_ddecl;
   Falcon::SymbolList *fal_symlist;
   Falcon::List *fal_genericList;



/* Line 214 of yacc.c  */
#line 412 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 264 of yacc.c  */
#line 424 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   6699

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  117
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  165
/* YYNRULES -- Number of rules.  */
#define YYNRULES  467
/* YYNRULES -- Number of states.  */
#define YYNSTATES  848

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   371

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    40,
      43,    47,    51,    55,    57,    59,    61,    65,    69,    73,
      76,    79,    84,    91,    93,    95,    97,    99,   101,   103,
     105,   107,   109,   111,   113,   115,   117,   119,   121,   123,
     125,   127,   131,   137,   141,   145,   146,   152,   155,   159,
     163,   167,   171,   172,   180,   184,   188,   189,   191,   192,
     199,   202,   206,   210,   214,   218,   219,   221,   222,   226,
     229,   233,   234,   239,   243,   247,   248,   251,   254,   258,
     261,   265,   269,   274,   279,   285,   289,   293,   294,   301,
     306,   311,   315,   316,   319,   322,   323,   326,   328,   330,
     332,   334,   338,   342,   346,   349,   353,   356,   360,   364,
     366,   367,   374,   378,   382,   383,   390,   394,   398,   399,
     406,   410,   414,   415,   422,   426,   430,   431,   434,   438,
     440,   441,   447,   448,   454,   455,   461,   462,   468,   469,
     470,   474,   475,   477,   480,   483,   486,   488,   492,   494,
     496,   498,   502,   504,   505,   512,   516,   520,   521,   524,
     528,   530,   531,   537,   538,   544,   545,   551,   552,   558,
     560,   564,   565,   567,   569,   573,   574,   581,   584,   588,
     589,   591,   593,   596,   599,   602,   607,   611,   617,   621,
     623,   627,   629,   631,   635,   639,   645,   648,   654,   655,
     663,   667,   673,   674,   681,   684,   685,   687,   691,   693,
     694,   695,   701,   702,   706,   709,   713,   716,   720,   724,
     728,   734,   740,   744,   747,   751,   755,   757,   761,   765,
     771,   777,   785,   793,   801,   809,   814,   819,   824,   829,
     836,   843,   847,   852,   857,   859,   863,   867,   871,   873,
     877,   881,   885,   889,   890,   898,   902,   905,   906,   910,
     911,   917,   918,   921,   923,   927,   930,   931,   934,   938,
     939,   942,   944,   946,   948,   950,   952,   954,   955,   963,
     969,   974,   979,   983,   987,   988,   991,   993,   995,   996,
    1004,  1005,  1008,  1010,  1015,  1017,  1020,  1022,  1024,  1025,
    1033,  1036,  1039,  1040,  1043,  1045,  1047,  1049,  1051,  1053,
    1054,  1059,  1061,  1063,  1066,  1070,  1074,  1076,  1079,  1083,
    1087,  1089,  1091,  1093,  1095,  1097,  1099,  1101,  1103,  1105,
    1107,  1109,  1111,  1113,  1115,  1117,  1119,  1121,  1123,  1124,
    1126,  1128,  1130,  1133,  1136,  1139,  1143,  1147,  1151,  1154,
    1158,  1163,  1168,  1173,  1178,  1183,  1188,  1193,  1198,  1203,
    1208,  1213,  1216,  1220,  1223,  1226,  1229,  1232,  1236,  1240,
    1244,  1248,  1252,  1256,  1260,  1263,  1267,  1271,  1275,  1278,
    1281,  1284,  1287,  1290,  1293,  1296,  1299,  1302,  1304,  1306,
    1308,  1310,  1312,  1314,  1317,  1319,  1324,  1330,  1334,  1336,
    1338,  1342,  1348,  1352,  1356,  1360,  1364,  1368,  1372,  1376,
    1380,  1384,  1388,  1392,  1396,  1400,  1405,  1410,  1416,  1424,
    1429,  1433,  1434,  1441,  1442,  1449,  1450,  1457,  1462,  1466,
    1469,  1472,  1475,  1478,  1479,  1486,  1492,  1498,  1503,  1507,
    1510,  1514,  1518,  1521,  1525,  1529,  1533,  1537,  1542,  1544,
    1548,  1550,  1554,  1555,  1557,  1559,  1563,  1567
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     118,     0,    -1,   119,    -1,    -1,   119,   120,    -1,   121,
      -1,    10,     3,    -1,    24,     1,     3,    -1,   123,    -1,
     220,    -1,   200,    -1,   223,    -1,   246,    -1,   241,    -1,
     124,    -1,   214,    -1,   215,    -1,   217,    -1,     4,    -1,
      98,     4,    -1,    38,     6,     3,    -1,    38,     7,     3,
      -1,    38,     1,     3,    -1,   125,    -1,   218,    -1,     3,
      -1,    45,     1,     3,    -1,    34,     1,     3,    -1,    32,
       1,     3,    -1,     1,     3,    -1,   261,     3,    -1,   277,
      76,   261,     3,    -1,   277,    76,   261,    79,   277,     3,
      -1,   127,    -1,   128,    -1,   132,    -1,   149,    -1,   164,
      -1,   179,    -1,   135,    -1,   146,    -1,   147,    -1,   190,
      -1,   199,    -1,   255,    -1,   251,    -1,   213,    -1,   155,
      -1,   156,    -1,   157,    -1,   258,    -1,   258,    76,   261,
      -1,   126,    79,   258,    76,   261,    -1,    11,   126,     3,
      -1,    11,     1,     3,    -1,    -1,   130,   129,   145,    10,
       3,    -1,   131,   124,    -1,    12,   261,     3,    -1,    12,
       1,     3,    -1,    12,   261,    44,    -1,    12,     1,    44,
      -1,    -1,    50,     3,   133,   145,    10,   134,     3,    -1,
      50,    44,   124,    -1,    50,     1,     3,    -1,    -1,   261,
      -1,    -1,   137,   136,   145,   139,    10,     3,    -1,   138,
     124,    -1,    16,   261,     3,    -1,    16,     1,     3,    -1,
      16,   261,    44,    -1,    16,     1,    44,    -1,    -1,   142,
      -1,    -1,   141,   140,   145,    -1,    17,     3,    -1,    17,
       1,     3,    -1,    -1,   144,   143,   145,   139,    -1,    18,
     261,     3,    -1,    18,     1,     3,    -1,    -1,   145,   124,
      -1,    13,     3,    -1,    13,     1,     3,    -1,    14,     3,
      -1,    14,    15,     3,    -1,    14,     1,     3,    -1,    19,
     280,    92,   261,    -1,    19,   258,    76,   151,    -1,    19,
     280,    92,     1,     3,    -1,    19,     1,     3,    -1,   148,
      44,   124,    -1,    -1,   148,     3,   150,   153,    10,     3,
      -1,   261,    78,   261,   152,    -1,   261,    78,   261,     1,
      -1,   261,    78,     1,    -1,    -1,    79,   261,    -1,    79,
       1,    -1,    -1,   154,   153,    -1,   124,    -1,   158,    -1,
     160,    -1,   162,    -1,    48,   261,     3,    -1,    48,     1,
       3,    -1,   104,   277,     3,    -1,   104,     3,    -1,    87,
     277,     3,    -1,    87,     3,    -1,   104,     1,     3,    -1,
      87,     1,     3,    -1,    55,    -1,    -1,    20,     3,   159,
     145,    10,     3,    -1,    20,    44,   124,    -1,    20,     1,
       3,    -1,    -1,    21,     3,   161,   145,    10,     3,    -1,
      21,    44,   124,    -1,    21,     1,     3,    -1,    -1,    22,
       3,   163,   145,    10,     3,    -1,    22,    44,   124,    -1,
      22,     1,     3,    -1,    -1,   166,   165,   167,   173,    10,
       3,    -1,    23,   261,     3,    -1,    23,     1,     3,    -1,
      -1,   167,   168,    -1,   167,     1,     3,    -1,     3,    -1,
      -1,    24,   177,     3,   169,   145,    -1,    -1,    24,   177,
      44,   170,   124,    -1,    -1,    24,     1,     3,   171,   145,
      -1,    -1,    24,     1,    44,   172,   124,    -1,    -1,    -1,
     175,   174,   176,    -1,    -1,    25,    -1,    25,     1,    -1,
       3,   145,    -1,    44,   124,    -1,   178,    -1,   177,    79,
     178,    -1,     8,    -1,   122,    -1,     7,    -1,   122,    78,
     122,    -1,     6,    -1,    -1,   181,   180,   182,   173,    10,
       3,    -1,    26,   261,     3,    -1,    26,     1,     3,    -1,
      -1,   182,   183,    -1,   182,     1,     3,    -1,     3,    -1,
      -1,    24,   188,     3,   184,   145,    -1,    -1,    24,   188,
      44,   185,   124,    -1,    -1,    24,     1,     3,   186,   145,
      -1,    -1,    24,     1,    44,   187,   124,    -1,   189,    -1,
     188,    79,   189,    -1,    -1,     4,    -1,     6,    -1,    29,
      44,   124,    -1,    -1,   192,   191,   145,   193,    10,     3,
      -1,    29,     3,    -1,    29,     1,     3,    -1,    -1,   194,
      -1,   195,    -1,   194,   195,    -1,   196,   145,    -1,    30,
       3,    -1,    30,    92,   258,     3,    -1,    30,   197,     3,
      -1,    30,   197,    92,   258,     3,    -1,    30,     1,     3,
      -1,   198,    -1,   197,    79,   198,    -1,     4,    -1,     6,
      -1,    31,   261,     3,    -1,    31,     1,     3,    -1,   201,
     208,   145,    10,     3,    -1,   203,   124,    -1,   205,    57,
     206,    56,     3,    -1,    -1,   205,    57,   206,     1,   202,
      56,     3,    -1,   205,     1,     3,    -1,   205,    57,   206,
      56,    44,    -1,    -1,   205,    57,     1,   204,    56,    44,
      -1,    45,     6,    -1,    -1,   207,    -1,   206,    79,   207,
      -1,     6,    -1,    -1,    -1,   211,   209,   145,    10,     3,
      -1,    -1,   212,   210,   124,    -1,    46,     3,    -1,    46,
       1,     3,    -1,    46,    44,    -1,    46,     1,    44,    -1,
      39,   263,     3,    -1,    39,     1,     3,    -1,    40,     6,
      76,   257,     3,    -1,    40,     6,    76,     1,     3,    -1,
      40,     1,     3,    -1,    41,     3,    -1,    41,   216,     3,
      -1,    41,     1,     3,    -1,     6,    -1,   216,    79,     6,
      -1,    42,   219,     3,    -1,    42,   219,    33,     6,     3,
      -1,    42,   219,    33,     7,     3,    -1,    42,   219,    33,
       6,    77,     6,     3,    -1,    42,   219,    33,     7,    77,
       6,     3,    -1,    42,   219,    33,     6,    92,     6,     3,
      -1,    42,   219,    33,     7,    92,     6,     3,    -1,    42,
       6,     1,     3,    -1,    42,   219,     1,     3,    -1,    42,
      33,     6,     3,    -1,    42,    33,     7,     3,    -1,    42,
      33,     6,    77,     6,     3,    -1,    42,    33,     7,    77,
       6,     3,    -1,    42,     1,     3,    -1,     6,    44,   257,
       3,    -1,     6,    44,     1,     3,    -1,     6,    -1,   219,
      79,     6,    -1,    43,   221,     3,    -1,    43,     1,     3,
      -1,   222,    -1,   221,    79,   222,    -1,     6,    76,     6,
      -1,     6,    76,     7,    -1,     6,    76,   122,    -1,    -1,
      32,     6,   224,   225,   232,    10,     3,    -1,   226,   228,
       3,    -1,     1,     3,    -1,    -1,    57,   206,    56,    -1,
      -1,    57,   206,     1,   227,    56,    -1,    -1,    33,   229,
      -1,   230,    -1,   229,    79,   230,    -1,     6,   231,    -1,
      -1,    57,    56,    -1,    57,   277,    56,    -1,    -1,   232,
     233,    -1,     3,    -1,   200,    -1,   236,    -1,   237,    -1,
     234,    -1,   218,    -1,    -1,    37,     3,   235,   208,   145,
      10,     3,    -1,    46,     6,    76,   257,     3,    -1,     6,
      76,   261,     3,    -1,   238,   239,    10,     3,    -1,    54,
       6,     3,    -1,    54,    37,     3,    -1,    -1,   239,   240,
      -1,     3,    -1,   200,    -1,    -1,    51,     6,   242,     3,
     243,    10,     3,    -1,    -1,   243,   244,    -1,     3,    -1,
       6,    76,   257,   245,    -1,   218,    -1,     6,   245,    -1,
       3,    -1,    79,    -1,    -1,    34,     6,   247,   248,   249,
      10,     3,    -1,   228,     3,    -1,     1,     3,    -1,    -1,
     249,   250,    -1,     3,    -1,   200,    -1,   236,    -1,   234,
      -1,   218,    -1,    -1,    36,   252,   253,     3,    -1,   254,
      -1,     1,    -1,   254,     1,    -1,   253,    79,   254,    -1,
     253,    79,     1,    -1,     6,    -1,    35,     3,    -1,    35,
     261,     3,    -1,    35,     1,     3,    -1,     8,    -1,     9,
      -1,    52,    -1,    53,    -1,     4,    -1,     5,    -1,     7,
      -1,     8,    -1,     9,    -1,    52,    -1,    53,    -1,   122,
      -1,     5,    -1,     7,    -1,     6,    -1,   258,    -1,    27,
      -1,    28,    -1,    -1,     3,    -1,   256,    -1,   259,    -1,
     113,     6,    -1,   113,     4,    -1,   113,    27,    -1,   113,
      60,     6,    -1,   113,    60,     4,    -1,   113,    60,    27,
      -1,    98,   261,    -1,     6,    64,   261,    -1,   261,    99,
     260,   261,    -1,   261,    98,   260,   261,    -1,   261,   102,
     260,   261,    -1,   261,   101,   260,   261,    -1,   261,   100,
     260,   261,    -1,   261,   103,   260,   261,    -1,   261,    97,
     260,   261,    -1,   261,    96,   260,   261,    -1,   261,    95,
     260,   261,    -1,   261,   105,   260,   261,    -1,   261,   104,
     260,   261,    -1,   111,   261,    -1,   261,    88,   261,    -1,
     261,   115,    -1,   115,   261,    -1,   261,   114,    -1,   114,
     261,    -1,   261,    89,   261,    -1,   261,    87,   261,    -1,
     261,    86,   261,    -1,   261,    85,   261,    -1,   261,    84,
     261,    -1,   261,    82,   261,    -1,   261,    81,   261,    -1,
      83,   261,    -1,   261,    92,   261,    -1,   261,    91,   261,
      -1,   261,    90,     6,    -1,   116,   258,    -1,   116,     4,
      -1,    94,   261,    -1,    93,   261,    -1,   110,   261,    -1,
     109,   261,    -1,   108,   261,    -1,   107,   261,    -1,   106,
     261,    -1,   265,    -1,   267,    -1,   271,    -1,   263,    -1,
     273,    -1,   275,    -1,   261,   262,    -1,   274,    -1,   261,
      59,   261,    58,    -1,   261,    59,   102,   261,    58,    -1,
     261,    60,     6,    -1,   276,    -1,   262,    -1,   261,    76,
     261,    -1,   261,    76,   261,    79,   277,    -1,   261,    75,
     261,    -1,   261,    74,   261,    -1,   261,    73,   261,    -1,
     261,    72,   261,    -1,   261,    71,   261,    -1,   261,    65,
     261,    -1,   261,    70,   261,    -1,   261,    69,   261,    -1,
     261,    68,   261,    -1,   261,    66,   261,    -1,   261,    67,
     261,    -1,    57,   261,    56,    -1,    59,    44,    58,    -1,
      59,   261,    44,    58,    -1,    59,    44,   261,    58,    -1,
      59,   261,    44,   261,    58,    -1,    59,   261,    44,   261,
      44,   261,    58,    -1,   261,    57,   277,    56,    -1,   261,
      57,    56,    -1,    -1,   261,    57,   277,     1,   264,    56,
      -1,    -1,    45,   266,   269,   208,   145,    10,    -1,    -1,
      61,   268,   270,   208,   145,    62,    -1,    57,   206,    56,
       3,    -1,    57,   206,     1,    -1,     1,     3,    -1,   206,
      63,    -1,   206,     1,    -1,     1,    63,    -1,    -1,    47,
     272,   269,   208,   145,    10,    -1,   261,    80,   261,    44,
     261,    -1,   261,    80,   261,    44,     1,    -1,   261,    80,
     261,     1,    -1,   261,    80,     1,    -1,    59,    58,    -1,
      59,   277,    58,    -1,    59,   277,     1,    -1,    49,    58,
      -1,    49,   278,    58,    -1,    49,   278,     1,    -1,    59,
      63,    58,    -1,    59,   281,    58,    -1,    59,   281,     1,
      58,    -1,   261,    -1,   277,    79,   261,    -1,   261,    -1,
     278,   279,   261,    -1,    -1,    79,    -1,   258,    -1,   280,
      79,   258,    -1,   261,    63,   261,    -1,   281,    79,   261,
      63,   261,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   198,   198,   201,   203,   207,   208,   209,   213,   214,
     215,   220,   225,   230,   235,   240,   241,   242,   246,   247,
     251,   257,   263,   270,   271,   272,   273,   274,   275,   276,
     281,   292,   298,   312,   313,   314,   315,   316,   317,   318,
     319,   320,   321,   322,   323,   324,   325,   326,   327,   328,
     332,   335,   341,   349,   351,   356,   356,   370,   378,   379,
     383,   384,   388,   388,   403,   409,   416,   417,   421,   421,
     436,   446,   447,   451,   452,   456,   458,   459,   459,   468,
     469,   474,   474,   486,   487,   490,   492,   498,   507,   515,
     525,   534,   542,   547,   555,   560,   569,   588,   587,   608,
     614,   618,   625,   626,   627,   630,   632,   636,   643,   644,
     645,   649,   662,   670,   674,   680,   686,   693,   698,   707,
     717,   717,   731,   740,   744,   744,   757,   766,   770,   770,
     786,   795,   799,   799,   816,   817,   824,   826,   827,   831,
     833,   832,   843,   843,   855,   855,   867,   867,   883,   886,
     885,   898,   899,   900,   903,   904,   910,   911,   915,   924,
     936,   947,   958,   979,   979,   996,   997,  1004,  1006,  1007,
    1011,  1013,  1012,  1023,  1023,  1036,  1036,  1048,  1048,  1066,
    1067,  1070,  1071,  1083,  1104,  1111,  1110,  1129,  1130,  1133,
    1135,  1139,  1140,  1144,  1149,  1167,  1187,  1197,  1208,  1216,
    1217,  1221,  1233,  1256,  1257,  1264,  1274,  1283,  1284,  1284,
    1288,  1292,  1293,  1293,  1300,  1403,  1405,  1406,  1410,  1425,
    1428,  1427,  1439,  1438,  1453,  1454,  1458,  1459,  1468,  1472,
    1480,  1490,  1495,  1507,  1516,  1523,  1531,  1536,  1544,  1549,
    1554,  1559,  1579,  1598,  1603,  1608,  1613,  1627,  1632,  1637,
    1642,  1647,  1656,  1661,  1668,  1674,  1686,  1691,  1699,  1700,
    1704,  1708,  1712,  1726,  1725,  1788,  1791,  1797,  1799,  1800,
    1800,  1806,  1808,  1812,  1813,  1817,  1841,  1842,  1843,  1850,
    1852,  1856,  1857,  1860,  1879,  1899,  1900,  1904,  1904,  1938,
    1960,  1988,  2000,  2008,  2021,  2023,  2028,  2029,  2041,  2040,
    2084,  2086,  2090,  2091,  2095,  2096,  2103,  2103,  2112,  2111,
    2178,  2179,  2185,  2187,  2191,  2192,  2195,  2214,  2215,  2224,
    2223,  2241,  2242,  2247,  2252,  2253,  2260,  2276,  2277,  2278,
    2286,  2287,  2288,  2289,  2290,  2291,  2292,  2296,  2297,  2298,
    2299,  2300,  2301,  2302,  2306,  2324,  2325,  2326,  2346,  2348,
    2352,  2353,  2354,  2355,  2356,  2357,  2358,  2359,  2360,  2361,
    2362,  2388,  2389,  2409,  2433,  2450,  2451,  2452,  2453,  2454,
    2455,  2456,  2457,  2458,  2459,  2460,  2461,  2462,  2463,  2464,
    2465,  2466,  2467,  2468,  2469,  2470,  2471,  2472,  2473,  2474,
    2475,  2476,  2477,  2478,  2479,  2480,  2481,  2482,  2483,  2484,
    2485,  2486,  2487,  2489,  2494,  2498,  2503,  2509,  2518,  2519,
    2521,  2526,  2533,  2534,  2535,  2536,  2537,  2538,  2539,  2540,
    2541,  2542,  2543,  2544,  2549,  2552,  2555,  2558,  2561,  2567,
    2573,  2578,  2578,  2588,  2587,  2631,  2630,  2682,  2683,  2687,
    2694,  2695,  2699,  2707,  2706,  2756,  2761,  2768,  2775,  2785,
    2786,  2790,  2798,  2799,  2803,  2812,  2813,  2814,  2822,  2823,
    2827,  2828,  2831,  2832,  2835,  2841,  2848,  2849
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "UNB", "END", "DEF", "WHILE", "BREAK", "CONTINUE",
  "DROPPING", "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST",
  "FORMIDDLE", "SWITCH", "CASE", "DEFAULT", "SELECT", "SELF", "FSELF",
  "TRY", "CATCH", "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL",
  "INIT", "LOAD", "LAUNCH", "CONST_KW", "EXPORT", "IMPORT", "DIRECTIVE",
  "COLON", "FUNCDECL", "STATIC", "INNERFUNC", "FORDOT", "LISTPAR", "LOOP",
  "ENUM", "TRUE_TOKEN", "FALSE_TOKEN", "STATE", "OUTER_STRING", "CLOSEPAR",
  "OPENPAR", "CLOSESQUARE", "OPENSQUARE", "DOT", "OPEN_GRAPH",
  "CLOSE_GRAPH", "ARROW", "VBAR", "ASSIGN_POW", "ASSIGN_SHL", "ASSIGN_SHR",
  "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD", "ASSIGN_DIV",
  "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD", "OP_EQ", "OP_AS", "OP_TO",
  "COMMA", "QUESTION", "OR", "AND", "NOT", "LE", "GE", "LT", "GT", "NEQ",
  "EEQ", "PROVIDES", "OP_NOTIN", "OP_IN", "DIESIS", "ATSIGN", "CAP_CAP",
  "VBAR_VBAR", "AMPER_AMPER", "MINUS", "PLUS", "PERCENT", "SLASH", "STAR",
  "POW", "SHR", "SHL", "CAP_XOROOB", "CAP_ISOOB", "CAP_DEOOB", "CAP_OOB",
  "CAP_EVAL", "TILDE", "NEG", "AMPER", "DECREMENT", "INCREMENT", "DOLLAR",
  "$accept", "input", "body", "line", "toplevel_statement",
  "INTNUM_WITH_MINUS", "load_statement", "statement", "base_statement",
  "assignment_def_list", "def_statement", "while_statement", "$@1",
  "while_decl", "while_short_decl", "loop_statement", "$@2",
  "loop_terminator", "if_statement", "$@3", "if_decl", "if_short_decl",
  "elif_or_else", "$@4", "else_decl", "elif_statement", "$@5", "elif_decl",
  "statement_list", "break_statement", "continue_statement",
  "forin_header", "forin_statement", "$@6", "for_to_expr",
  "for_to_step_clause", "forin_statement_list", "forin_statement_elem",
  "fordot_statement", "self_print_statement", "outer_print_statement",
  "first_loop_block", "$@7", "last_loop_block", "$@8", "middle_loop_block",
  "$@9", "switch_statement", "$@10", "switch_decl", "case_list",
  "case_statement", "$@11", "$@12", "$@13", "$@14", "default_statement",
  "$@15", "default_decl", "default_body", "case_expression_list",
  "case_element", "select_statement", "$@16", "select_decl",
  "selcase_list", "selcase_statement", "$@17", "$@18", "$@19", "$@20",
  "selcase_expression_list", "selcase_element", "try_statement", "$@21",
  "try_decl", "catch_statements", "catch_list", "catch_statement",
  "catch_decl", "catchcase_element_list", "catchcase_element",
  "raise_statement", "func_statement", "func_decl", "$@22",
  "func_decl_short", "$@23", "func_begin", "param_list", "param_symbol",
  "static_block", "$@24", "$@25", "static_decl", "static_short_decl",
  "launch_statement", "const_statement", "export_statement",
  "export_symbol_list", "import_statement", "attribute_statement",
  "import_symbol_list", "directive_statement", "directive_pair_list",
  "directive_pair", "class_decl", "$@26", "class_def_inner",
  "class_param_list", "$@27", "from_clause", "inherit_list",
  "inherit_token", "inherit_call", "class_statement_list",
  "class_statement", "init_decl", "$@28", "property_decl", "state_decl",
  "state_heading", "state_statement_list", "state_statement",
  "enum_statement", "$@29", "enum_statement_list", "enum_item_decl",
  "enum_item_terminator", "object_decl", "$@30", "object_decl_inner",
  "object_statement_list", "object_statement", "global_statement", "$@31",
  "global_symbol_list", "globalized_symbol", "return_statement",
  "const_atom_non_minus", "const_atom", "atomic_symbol", "var_atom",
  "OPT_EOL", "expression", "range_decl", "func_call", "$@32",
  "nameless_func", "$@33", "nameless_block", "$@34",
  "nameless_func_decl_inner", "nameless_block_decl_inner", "innerfunc",
  "$@35", "iif_expr", "array_decl", "dotarray_decl", "dict_decl",
  "expression_list", "listpar_expression_list", "listpar_comma",
  "symbol_list", "expression_pair_list", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   117,   118,   119,   119,   120,   120,   120,   121,   121,
     121,   121,   121,   121,   121,   121,   121,   121,   122,   122,
     123,   123,   123,   124,   124,   124,   124,   124,   124,   124,
     125,   125,   125,   125,   125,   125,   125,   125,   125,   125,
     125,   125,   125,   125,   125,   125,   125,   125,   125,   125,
     126,   126,   126,   127,   127,   129,   128,   128,   130,   130,
     131,   131,   133,   132,   132,   132,   134,   134,   136,   135,
     135,   137,   137,   138,   138,   139,   139,   140,   139,   141,
     141,   143,   142,   144,   144,   145,   145,   146,   146,   147,
     147,   147,   148,   148,   148,   148,   149,   150,   149,   151,
     151,   151,   152,   152,   152,   153,   153,   154,   154,   154,
     154,   155,   155,   156,   156,   156,   156,   156,   156,   157,
     159,   158,   158,   158,   161,   160,   160,   160,   163,   162,
     162,   162,   165,   164,   166,   166,   167,   167,   167,   168,
     169,   168,   170,   168,   171,   168,   172,   168,   173,   174,
     173,   175,   175,   175,   176,   176,   177,   177,   178,   178,
     178,   178,   178,   180,   179,   181,   181,   182,   182,   182,
     183,   184,   183,   185,   183,   186,   183,   187,   183,   188,
     188,   189,   189,   189,   190,   191,   190,   192,   192,   193,
     193,   194,   194,   195,   196,   196,   196,   196,   196,   197,
     197,   198,   198,   199,   199,   200,   200,   201,   202,   201,
     201,   203,   204,   203,   205,   206,   206,   206,   207,   208,
     209,   208,   210,   208,   211,   211,   212,   212,   213,   213,
     214,   214,   214,   215,   215,   215,   216,   216,   217,   217,
     217,   217,   217,   217,   217,   217,   217,   217,   217,   217,
     217,   217,   218,   218,   219,   219,   220,   220,   221,   221,
     222,   222,   222,   224,   223,   225,   225,   226,   226,   227,
     226,   228,   228,   229,   229,   230,   231,   231,   231,   232,
     232,   233,   233,   233,   233,   233,   233,   235,   234,   236,
     236,   237,   238,   238,   239,   239,   240,   240,   242,   241,
     243,   243,   244,   244,   244,   244,   245,   245,   247,   246,
     248,   248,   249,   249,   250,   250,   250,   250,   250,   252,
     251,   253,   253,   253,   253,   253,   254,   255,   255,   255,
     256,   256,   256,   256,   256,   256,   256,   257,   257,   257,
     257,   257,   257,   257,   258,   259,   259,   259,   260,   260,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   262,   262,   262,   262,   262,   263,
     263,   264,   263,   266,   265,   268,   267,   269,   269,   269,
     270,   270,   270,   272,   271,   273,   273,   273,   273,   274,
     274,   274,   275,   275,   275,   276,   276,   276,   277,   277,
     278,   278,   279,   279,   280,   280,   281,   281
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       3,     3,     3,     1,     1,     1,     3,     3,     3,     2,
       2,     4,     6,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     5,     3,     3,     0,     5,     2,     3,     3,
       3,     3,     0,     7,     3,     3,     0,     1,     0,     6,
       2,     3,     3,     3,     3,     0,     1,     0,     3,     2,
       3,     0,     4,     3,     3,     0,     2,     2,     3,     2,
       3,     3,     4,     4,     5,     3,     3,     0,     6,     4,
       4,     3,     0,     2,     2,     0,     2,     1,     1,     1,
       1,     3,     3,     3,     2,     3,     2,     3,     3,     1,
       0,     6,     3,     3,     0,     6,     3,     3,     0,     6,
       3,     3,     0,     6,     3,     3,     0,     2,     3,     1,
       0,     5,     0,     5,     0,     5,     0,     5,     0,     0,
       3,     0,     1,     2,     2,     2,     1,     3,     1,     1,
       1,     3,     1,     0,     6,     3,     3,     0,     2,     3,
       1,     0,     5,     0,     5,     0,     5,     0,     5,     1,
       3,     0,     1,     1,     3,     0,     6,     2,     3,     0,
       1,     1,     2,     2,     2,     4,     3,     5,     3,     1,
       3,     1,     1,     3,     3,     5,     2,     5,     0,     7,
       3,     5,     0,     6,     2,     0,     1,     3,     1,     0,
       0,     5,     0,     3,     2,     3,     2,     3,     3,     3,
       5,     5,     3,     2,     3,     3,     1,     3,     3,     5,
       5,     7,     7,     7,     7,     4,     4,     4,     4,     6,
       6,     3,     4,     4,     1,     3,     3,     3,     1,     3,
       3,     3,     3,     0,     7,     3,     2,     0,     3,     0,
       5,     0,     2,     1,     3,     2,     0,     2,     3,     0,
       2,     1,     1,     1,     1,     1,     1,     0,     7,     5,
       4,     4,     3,     3,     0,     2,     1,     1,     0,     7,
       0,     2,     1,     4,     1,     2,     1,     1,     0,     7,
       2,     2,     0,     2,     1,     1,     1,     1,     1,     0,
       4,     1,     1,     2,     3,     3,     1,     2,     3,     3,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     0,     1,
       1,     1,     2,     2,     2,     3,     3,     3,     2,     3,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     2,     3,     2,     2,     2,     2,     3,     3,     3,
       3,     3,     3,     3,     2,     3,     3,     3,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     1,     1,     1,
       1,     1,     1,     2,     1,     4,     5,     3,     1,     1,
       3,     5,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     4,     4,     5,     7,     4,
       3,     0,     6,     0,     6,     0,     6,     4,     3,     2,
       2,     2,     2,     0,     6,     5,     5,     4,     3,     2,
       3,     3,     2,     3,     3,     3,     3,     4,     1,     3,
       1,     3,     0,     1,     1,     3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    25,   334,   335,   344,   336,
     330,   331,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   346,   347,     0,     0,     0,     0,     0,   319,
       0,     0,     0,     0,     0,     0,     0,   443,     0,     0,
       0,     0,   332,   333,   119,     0,     0,   435,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     4,     5,     8,    14,    23,    33,
      34,    55,     0,    35,    39,    68,     0,    40,    41,     0,
      36,    47,    48,    49,    37,   132,    38,   163,    42,   185,
      43,    10,   219,     0,     0,    46,    15,    16,    17,    24,
       9,    11,    13,    12,    45,    44,   350,   345,   351,   458,
     409,   400,   397,   398,   399,   401,   404,   402,   408,     0,
      29,     0,     0,     6,     0,   344,     0,    50,     0,   344,
     433,     0,     0,    87,     0,    89,     0,     0,     0,     0,
     464,     0,     0,     0,     0,     0,     0,     0,   187,     0,
       0,     0,     0,   263,     0,   308,     0,   327,     0,     0,
       0,     0,     0,     0,     0,   400,     0,     0,     0,   233,
     236,     0,     0,     0,     0,     0,     0,     0,     0,   258,
       0,   214,     0,     0,     0,     0,   452,   460,     0,     0,
      62,     0,   298,     0,     0,   449,     0,   458,     0,     0,
       0,   384,     0,   116,   458,     0,   391,   390,   358,     0,
     114,     0,   396,   395,   394,   393,   392,   371,   353,   352,
     354,     0,   376,   374,   389,   388,    85,     0,     0,     0,
      57,    85,    70,    97,     0,   136,   167,    85,     0,    85,
     220,   222,   206,     0,     0,    30,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   348,   348,   348,   348,   348,   348,   348,
     348,   348,   348,   348,   375,   373,   403,     0,     0,     0,
      18,   342,   343,   337,   338,   339,   340,     0,   341,     0,
     359,    54,    53,     0,     0,    59,    61,    58,    60,    88,
      91,    90,    72,    74,    71,    73,    95,     0,     0,     0,
     135,   134,     7,   166,   165,   188,   184,   204,   203,    28,
       0,    27,     0,   329,   328,   322,   326,     0,     0,    22,
      20,    21,   229,   228,   232,     0,   235,   234,     0,   251,
       0,     0,     0,     0,   238,     0,     0,   257,     0,   256,
       0,    26,     0,   215,   219,   219,   112,   111,   454,   453,
     463,     0,    65,    85,    64,     0,   423,   424,     0,   455,
       0,     0,   451,   450,     0,   456,     0,     0,   218,     0,
     216,   219,   118,   115,   117,   113,   356,   355,   357,     0,
       0,     0,    96,     0,     0,     0,     0,   224,   226,     0,
      85,     0,   210,   212,     0,   430,     0,     0,     0,   407,
     417,   421,   422,   420,   419,   418,   416,   415,   414,   413,
     412,   410,   448,     0,   383,   382,   381,   380,   379,   378,
     372,   377,   387,   386,   385,   349,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   459,   253,
      19,   252,     0,    51,    93,     0,   465,     0,    92,     0,
     215,   279,   271,     0,     0,     0,   312,   320,     0,   323,
       0,     0,   237,   245,   247,     0,   248,     0,   246,     0,
       0,   255,   260,   261,   262,   259,   439,     0,    85,    85,
     461,     0,   300,   426,   425,     0,   466,   457,     0,   442,
     441,   440,     0,    85,     0,    86,     0,     0,     0,    77,
      76,    81,     0,     0,     0,   107,     0,     0,   108,   109,
     110,     0,   139,     0,     0,   137,     0,   149,     0,   170,
       0,     0,   168,     0,     0,   190,   191,    85,   225,   227,
       0,     0,   223,     0,   208,     0,   431,   429,     0,   405,
       0,   447,     0,   368,   367,   366,   361,   360,   364,   363,
     362,   365,   370,   369,    31,     0,     0,     0,    94,   266,
       0,     0,     0,   311,   276,   272,   273,   310,     0,   325,
     324,   231,   230,     0,     0,   239,     0,     0,   240,     0,
       0,   438,     0,     0,     0,    66,     0,     0,   427,     0,
     217,     0,    56,     0,    79,     0,     0,     0,    85,    85,
       0,   120,     0,     0,   124,     0,     0,   128,     0,     0,
     106,   138,     0,   162,   160,   158,   159,     0,   156,   153,
       0,     0,   169,     0,   182,   183,     0,   179,     0,     0,
     194,   201,   202,     0,     0,   199,     0,   192,     0,   205,
       0,     0,     0,   207,   211,     0,   406,   411,   446,   445,
       0,    52,   101,     0,   269,   268,   281,     0,     0,     0,
       0,     0,     0,   282,   286,   280,   285,   283,   284,   294,
     265,     0,   275,     0,   314,     0,   315,   318,   317,   316,
     313,   249,   250,     0,     0,     0,     0,   437,   434,   444,
       0,    67,   302,     0,     0,   304,   301,     0,   467,   436,
      80,    84,    83,    69,     0,     0,   123,    85,   122,   127,
      85,   126,   131,    85,   130,    98,   144,   146,     0,   140,
     142,     0,   133,    85,     0,   150,   175,   177,   171,   173,
     181,   164,   198,     0,   196,     0,     0,   186,   221,   213,
       0,   432,    32,   100,     0,    99,     0,     0,   264,   287,
       0,     0,     0,     0,   277,     0,   274,   309,   241,   243,
     242,   244,    63,   306,     0,   307,   305,   299,   428,    82,
       0,     0,     0,    85,     0,   161,    85,     0,   157,     0,
     155,    85,     0,    85,     0,   180,   195,   200,     0,   209,
     104,   103,   270,     0,   219,     0,   292,   293,   296,     0,
     297,   295,   278,     0,     0,     0,     0,     0,   147,     0,
     143,     0,   178,     0,   174,   197,   290,    85,     0,   291,
     303,   121,   125,   129,     0,   289,     0,   288
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    64,    65,   298,    66,   515,    68,   126,
      69,    70,   226,    71,    72,    73,   373,   710,    74,   231,
      75,    76,   518,   618,   519,   520,   619,   521,   399,    77,
      78,    79,    80,   401,   464,   765,   526,   527,    81,    82,
      83,   528,   727,   529,   730,   530,   733,    84,   235,    85,
     403,   535,   796,   797,   793,   794,   536,   641,   537,   745,
     637,   638,    86,   236,    87,   404,   542,   803,   804,   801,
     802,   646,   647,    88,   237,    89,   544,   545,   546,   547,
     654,   655,    90,    91,    92,   662,    93,   553,    94,   389,
     390,   239,   410,   411,   240,   241,    95,    96,    97,   171,
      98,    99,   175,   100,   178,   179,   101,   330,   471,   472,
     766,   475,   585,   586,   692,   581,   685,   686,   814,   687,
     688,   689,   773,   821,   102,   375,   606,   716,   786,   103,
     332,   476,   588,   700,   104,   159,   337,   338,   105,   106,
     299,   107,   108,   446,   109,   110,   111,   665,   112,   182,
     113,   200,   364,   391,   114,   183,   115,   116,   117,   118,
     119,   188,   371,   141,   199
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -566
static const yytype_int16 yypact[] =
{
    -566,    27,   916,  -566,    68,  -566,  -566,  -566,   334,  -566,
    -566,  -566,   101,   316,  3640,   172,   513,  3712,   332,  3784,
      88,  3856,  -566,  -566,   293,  3928,   542,   554,  3424,  -566,
     504,  4000,   573,   574,   297,   582,   322,  -566,  4072,  5485,
     313,   134,  -566,  -566,  -566,  5845,  5340,  -566,  5845,  3496,
    5845,  5845,  5845,  3568,  5845,  5845,  5845,  5845,  5845,  5845,
     325,  5845,  5845,   198,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,  3352,  -566,  -566,  -566,  3352,  -566,  -566,   281,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,   103,  3352,    33,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  4878,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,   -23,
    -566,    50,  5845,  -566,   183,  -566,    90,   122,   350,   148,
    -566,  4701,   216,  -566,   223,  -566,   296,   353,  4775,   307,
     279,   -14,   338,  4929,   377,   397,  4981,   410,  -566,  3352,
     419,  5032,   429,  -566,   450,  -566,   500,  -566,  5084,   583,
     510,   512,   519,   520,  6388,   527,   530,   420,   538,  -566,
    -566,    92,   558,   192,    56,   260,   564,   494,    98,  -566,
     578,  -566,    49,    49,   579,  5135,  -566,  6388,   493,   590,
    -566,  3352,  -566,  6080,  5557,  -566,   349,  5905,    71,    86,
      82,  6584,   592,  -566,  6388,   124,   565,   565,   291,   594,
    -566,   189,   291,   291,   291,   291,   291,   291,  -566,  -566,
    -566,   284,   291,   291,  -566,  -566,  -566,   567,   597,   106,
    -566,  -566,  -566,  -566,  3352,  -566,  -566,  -566,   423,  -566,
    -566,  -566,  -566,   618,    13,  -566,  5629,  5413,   621,  5845,
    5845,  5845,  5845,  5845,  5845,  5845,  5845,  5845,  5845,  5845,
    5845,  4144,  5845,  5845,  5845,  5845,  5845,  5845,  5845,  5845,
     622,  5845,  5845,   626,   626,   626,   626,   626,   626,   626,
     626,   626,   626,   626,  -566,  -566,  -566,  5845,  5845,   627,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,   628,  -566,   630,
    6388,  -566,  -566,   625,  5845,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  5845,   625,  4216,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
     312,  -566,   392,  -566,  -566,  -566,  -566,   193,    21,  -566,
    -566,  -566,  -566,  -566,  -566,    72,  -566,  -566,   629,  -566,
     631,   263,   266,   633,  -566,   453,   632,  -566,    24,  -566,
     634,  -566,   636,   635,   103,   103,  -566,  -566,  -566,  -566,
    -566,  5845,  -566,  -566,  -566,   641,  -566,  -566,  6131,  -566,
    5701,  5845,  -566,  -566,   587,  -566,  5845,   585,  -566,    74,
    -566,   103,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  1844,
    1496,  1032,  -566,   482,   534,  1612,   374,  -566,  -566,  1960,
    -566,  3352,  -566,  -566,   208,  -566,   229,  5845,  5967,  -566,
    6388,  6388,  6388,  6388,  6388,  6388,  6388,  6388,  6388,  6388,
    6388,   787,  -566,  4627,  6535,  6584,  5817,  5817,  5817,  5817,
    5817,  5817,  -566,   565,   565,  -566,  5845,  5845,  5845,  5845,
    5845,  5845,  5845,  5845,  5845,  5845,  5845,  4826,  6437,  -566,
    -566,  -566,   561,  6388,  -566,  6182,  -566,   643,  6388,   644,
     635,  -566,   616,   647,   645,   649,  -566,  -566,   584,  -566,
     650,   651,  -566,  -566,  -566,   652,  -566,   653,  -566,    95,
     185,  -566,  -566,  -566,  -566,  -566,  -566,   230,  -566,  -566,
    6388,  2076,  -566,  -566,  -566,  6029,  6388,  -566,  6235,  -566,
    -566,  -566,   635,  -566,   654,  -566,   331,  4288,   646,  -566,
    -566,  -566,   446,   473,   483,  -566,   661,  1032,  -566,  -566,
    -566,   669,  -566,    60,   488,  -566,   663,  -566,   671,  -566,
     275,   665,  -566,    93,   672,   667,  -566,  -566,  -566,  -566,
     678,  2192,  -566,   655,  -566,   407,  -566,  -566,  6286,  -566,
    5845,  -566,  4360,   679,   679,   261,   287,   287,   267,   267,
     267,   404,   291,   291,  -566,  5845,  5845,  4432,  -566,  -566,
     256,   411,   681,  -566,   656,   637,  -566,  -566,   409,  -566,
    -566,  -566,  -566,   704,   707,  -566,   706,   708,  -566,   709,
     711,  -566,   715,  2308,  2424,  5845,   563,  5845,  -566,  5845,
    -566,  2540,  -566,   716,  -566,   717,  5187,   718,  -566,  -566,
     721,  -566,  3352,   722,  -566,  3352,   723,  -566,  3352,   724,
    -566,  -566,   490,  -566,  -566,  -566,   657,   221,  -566,  -566,
     726,   492,  -566,   509,  -566,  -566,   257,  -566,   728,   729,
    -566,  -566,  -566,   625,   188,  -566,   731,  -566,  1728,  -566,
     734,   696,   685,  -566,  -566,   687,  -566,   668,  -566,  6486,
     203,  6388,  -566,  4565,  -566,  -566,  -566,   123,   741,   743,
     742,   744,    23,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  5773,  -566,   645,  -566,   746,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,   749,   750,   752,   753,  -566,  -566,  -566,
     754,  6388,  -566,   226,   755,  -566,  -566,  6337,  6388,  -566,
    -566,  -566,  -566,  -566,  2656,  1496,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,     7,  -566,
    -566,    78,  -566,  -566,  3352,  -566,  -566,  -566,  -566,  -566,
     424,  -566,  -566,   756,  -566,   525,   625,  -566,  -566,  -566,
     757,  -566,  -566,  -566,  4504,  -566,   705,  5845,  -566,  -566,
     686,   760,   761,   339,  -566,   118,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,   175,  -566,  -566,  -566,  -566,  -566,
    2772,  2888,  3004,  -566,  3352,  -566,  -566,  3352,  -566,  3120,
    -566,  -566,  3352,  -566,  3352,  -566,  -566,  -566,   763,  -566,
    -566,  6388,  -566,  5238,   103,   175,  -566,  -566,  -566,   764,
    -566,  -566,  -566,   204,   765,   766,   769,   107,  -566,  1148,
    -566,  1264,  -566,  1380,  -566,  -566,  -566,  -566,   771,  -566,
    -566,  -566,  -566,  -566,  3236,  -566,   772,  -566
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -566,  -566,  -566,  -566,  -566,  -355,  -566,    -2,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,    61,  -566,  -566,  -566,  -566,  -566,    58,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,   262,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,   383,  -566,  -566,  -566,
    -566,    55,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,    40,  -566,  -566,  -566,  -566,  -566,   252,  -566,
    -566,    43,  -566,  -565,  -566,  -566,  -566,  -566,  -566,  -235,
     292,  -344,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -161,  -566,  -566,  -566,   439,  -566,  -566,  -566,  -566,
    -566,   333,  -566,   110,  -566,  -566,  -566,   218,  -566,   219,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,   -15,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,   335,  -566,  -566,
    -340,   -11,  -566,   336,   -13,   265,   778,  -566,  -566,  -566,
    -566,  -566,   638,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
     -36,  -566,  -566,  -566,  -566
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -463
static const yytype_int16 yytable[] =
{
      67,   131,   127,   494,   138,   481,   143,   140,   146,   414,
     198,   290,   151,   205,   413,   158,   683,   211,   164,   388,
     498,   499,   479,   696,  -321,   185,   187,     3,   290,   771,
     492,   493,   193,   197,   243,   201,   204,   206,   207,   208,
     204,   212,   213,   214,   215,   216,   217,   513,   222,   223,
     362,   289,   225,   287,   290,   291,   288,   292,   293,   294,
     772,   632,   351,   352,   290,   318,   633,   634,   635,  -215,
     230,   120,   382,   480,   232,   510,   290,   291,   319,   292,
     293,   294,   290,   387,   633,   634,   635,   384,   388,   144,
     244,   242,  -215,   302,   649,   347,   650,   651,   595,   652,
    -321,   359,   295,   296,   123,   297,   363,   180,     4,   300,
       5,     6,     7,     8,     9,    10,    11,  -145,    13,    14,
      15,    16,   297,    17,   295,   296,    18,   393,   497,   383,
      19,  -145,  -145,    21,    22,    23,    24,   511,    25,   227,
     192,   228,    28,    29,   385,  -215,    31,   326,   297,   238,
     288,  -145,   229,   512,    37,    38,    39,    40,   297,    42,
      43,  -215,    44,  -433,    45,   386,    46,   121,    47,   303,
     297,   348,   596,   132,   822,   133,   297,   360,   636,   290,
     291,   378,   292,   293,   294,   653,   301,   597,   598,   374,
      48,   754,   395,   350,    49,  -254,   477,   288,   304,   767,
      50,    51,   224,   288,   125,    52,   762,   783,   820,   554,
     416,    53,   122,    54,    55,    56,    57,    58,    59,   309,
      60,    61,    62,    63,   739,  -254,   310,   295,   296,   783,
     556,   601,   402,   204,   418,   580,   420,   421,   422,   423,
     424,   425,   426,   427,   428,   429,   430,   431,   433,   434,
     435,   436,   437,   438,   439,   440,   441,   674,   443,   444,
     748,   353,   599,   354,   555,   740,   484,   755,   288,   486,
     121,  -254,   478,   297,   457,   458,   643,   600,  -181,   644,
     756,   645,   288,   785,   233,   557,   602,   512,   396,   400,
     397,   463,   462,   355,   147,   405,   148,   409,   172,   311,
     741,   749,   784,   173,   465,   785,   468,   466,   288,   512,
     316,   398,   675,   469,   189,  -267,   190,   124,   246,  -181,
     247,   248,   125,   180,   246,   234,   247,   248,   181,   218,
     174,   219,   613,   139,   614,   512,   750,   149,   125,   356,
     485,   320,   818,   487,   246,  -267,   247,   248,   246,   819,
     247,   248,   220,   305,  -181,   317,   312,   191,   500,   276,
     277,   278,   279,   280,   281,   282,   283,   505,   506,   470,
     281,   282,   283,   508,   286,   284,   285,   548,   121,  -433,
     322,   284,   285,   795,   680,   221,   636,   278,   279,   280,
     281,   282,   283,   473,   306,  -271,   286,   313,   122,   525,
     323,   284,   285,   286,   558,   284,   285,   379,   286,   552,
     663,   286,   694,   325,   676,   677,   286,   677,   549,   695,
     684,   678,   327,   286,   406,   474,   407,   697,   644,   286,
     645,   501,   329,   563,   564,   565,   566,   567,   568,   569,
     570,   571,   572,   573,   823,   715,   679,   620,   679,   621,
     286,   664,   286,   331,   680,   681,   680,   681,   286,   489,
     490,   246,   286,   247,   248,   682,   286,   408,   551,   286,
     837,   286,   286,   286,   623,   838,   624,   286,   286,   286,
     286,   286,   286,   531,   626,   532,   627,   286,   286,   639,
     622,  -152,  -148,   736,   368,   743,   345,  -462,  -462,  -462,
    -462,  -462,  -462,   333,   616,   160,   533,   534,   282,   283,
     161,   162,   746,   339,   134,   340,   135,   625,   284,   285,
    -462,  -462,   341,   342,   667,   525,  -151,   628,   136,   651,
     343,   652,  -152,   344,   737,   538,   744,   539,  -462,   670,
    -462,   346,  -462,   152,  -148,  -462,  -462,   204,   153,   669,
    -462,   369,  -462,   747,  -462,   154,   603,   604,   540,   534,
     155,   349,   204,   671,   673,   286,   712,   357,   152,   713,
     358,   611,   370,   714,   166,   168,  -462,   169,  -151,   167,
     170,   361,   366,   176,   335,   589,  -462,  -462,   177,   336,
     336,  -462,   711,   372,   717,   392,   718,   394,   154,  -462,
    -462,  -462,  -462,  -462,  -462,   658,  -462,  -462,  -462,  -462,
     447,   448,   449,   450,   451,   452,   453,   454,   455,   456,
     728,   412,   246,   731,   247,   248,   734,   419,   442,   445,
     459,   125,   460,   461,   483,   482,   488,   576,   491,   496,
     177,   388,   753,   286,   502,   507,   578,   579,   509,   474,
     583,   584,   587,   591,   592,   775,   617,   612,   593,   594,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   629,   631,   640,   642,   648,   724,   725,   204,   284,
     285,   659,   656,   286,   690,   286,   286,   286,   286,   286,
     286,   286,   286,   286,   286,   286,   286,   543,   286,   286,
     286,   286,   286,   286,   286,   286,   286,   701,   286,   286,
     702,   661,   703,   691,   704,   705,   693,   706,   707,   720,
     721,   723,   286,   286,   726,   729,   732,   735,   286,   742,
     286,   751,   752,   286,   757,   738,   246,   758,   247,   248,
     759,   760,   800,   761,   768,   808,   769,   288,   181,   777,
     770,   811,   778,   779,   813,   780,   781,   782,   787,   806,
     809,   812,   815,   816,   817,   286,   835,   839,   841,   842,
     286,   286,   843,   286,   845,   847,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   790,   789,   541,   791,   630,
     805,   792,   828,   284,   285,   830,   798,   657,   807,   495,
     832,   799,   834,   776,   610,   582,   698,   699,   840,   165,
       0,     0,     0,   590,     0,     0,     0,     0,     0,     0,
       0,   365,     0,   286,     0,     0,     0,     0,   286,   286,
     286,   286,   286,   286,   286,   286,   286,   286,   286,     0,
       0,     0,     0,     0,   246,     0,   247,   248,     0,     0,
       0,   827,     0,     0,   829,     0,     0,     0,     0,   831,
       0,   833,     0,   260,     0,     0,   560,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,   286,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,     0,     0,   844,     0,     0,     0,     0,
       0,   284,   285,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    -2,     4,     0,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,     0,    17,     0,   286,    18,   286,     0,   286,    19,
      20,     0,    21,    22,    23,    24,     0,    25,    26,     0,
      27,    28,    29,     0,    30,    31,    32,    33,    34,    35,
       0,    36,     0,    37,    38,    39,    40,    41,    42,    43,
       0,    44,     0,    45,     0,    46,   286,    47,     0,     0,
       0,     0,   286,   286,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,    50,
      51,     0,     0,     0,    52,     0,     0,     0,     0,     0,
      53,     0,    54,    55,    56,    57,    58,    59,     0,    60,
      61,    62,    63,     4,     0,     5,     6,     7,     8,     9,
      10,    11,  -105,    13,    14,    15,    16,     0,    17,     0,
       0,    18,   522,   523,   524,    19,     0,     0,    21,    22,
      23,    24,     0,    25,   227,     0,   228,    28,    29,     0,
       0,    31,     0,     0,     0,     0,   286,   229,   286,    37,
      38,    39,    40,     0,    42,    43,     0,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,    50,    51,     0,     0,     0,
      52,     0,     0,     0,     0,     0,    53,     0,    54,    55,
      56,    57,    58,    59,     0,    60,    61,    62,    63,     4,
       0,     5,     6,     7,     8,     9,    10,    11,  -141,    13,
      14,    15,    16,     0,    17,     0,     0,    18,     0,     0,
       0,    19,  -141,  -141,    21,    22,    23,    24,     0,    25,
     227,     0,   228,    28,    29,     0,     0,    31,     0,     0,
       0,     0,  -141,   229,     0,    37,    38,    39,    40,     0,
      42,    43,     0,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,    50,    51,     0,     0,     0,    52,     0,     0,     0,
       0,     0,    53,     0,    54,    55,    56,    57,    58,    59,
       0,    60,    61,    62,    63,     4,     0,     5,     6,     7,
       8,     9,    10,    11,  -176,    13,    14,    15,    16,     0,
      17,     0,     0,    18,     0,     0,     0,    19,  -176,  -176,
      21,    22,    23,    24,     0,    25,   227,     0,   228,    28,
      29,     0,     0,    31,     0,     0,     0,     0,  -176,   229,
       0,    37,    38,    39,    40,     0,    42,    43,     0,    44,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,    50,    51,     0,
       0,     0,    52,     0,     0,     0,     0,     0,    53,     0,
      54,    55,    56,    57,    58,    59,     0,    60,    61,    62,
      63,     4,     0,     5,     6,     7,     8,     9,    10,    11,
    -172,    13,    14,    15,    16,     0,    17,     0,     0,    18,
       0,     0,     0,    19,  -172,  -172,    21,    22,    23,    24,
       0,    25,   227,     0,   228,    28,    29,     0,     0,    31,
       0,     0,     0,     0,  -172,   229,     0,    37,    38,    39,
      40,     0,    42,    43,     0,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,    50,    51,     0,     0,     0,    52,     0,
       0,     0,     0,     0,    53,     0,    54,    55,    56,    57,
      58,    59,     0,    60,    61,    62,    63,     4,     0,     5,
       6,     7,     8,     9,    10,    11,   -75,    13,    14,    15,
      16,     0,    17,   516,   517,    18,     0,     0,     0,    19,
       0,     0,    21,    22,    23,    24,     0,    25,   227,     0,
     228,    28,    29,     0,     0,    31,     0,     0,     0,     0,
       0,   229,     0,    37,    38,    39,    40,     0,    42,    43,
       0,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,    50,
      51,     0,     0,     0,    52,     0,     0,     0,     0,     0,
      53,     0,    54,    55,    56,    57,    58,    59,     0,    60,
      61,    62,    63,     4,     0,     5,     6,     7,     8,     9,
      10,    11,  -189,    13,    14,    15,    16,     0,    17,     0,
       0,    18,     0,     0,     0,    19,     0,     0,    21,    22,
      23,    24,   543,    25,   227,     0,   228,    28,    29,     0,
       0,    31,     0,     0,     0,     0,     0,   229,     0,    37,
      38,    39,    40,     0,    42,    43,     0,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,    50,    51,     0,     0,     0,
      52,     0,     0,     0,     0,     0,    53,     0,    54,    55,
      56,    57,    58,    59,     0,    60,    61,    62,    63,     4,
       0,     5,     6,     7,     8,     9,    10,    11,  -193,    13,
      14,    15,    16,     0,    17,     0,     0,    18,     0,     0,
       0,    19,     0,     0,    21,    22,    23,    24,  -193,    25,
     227,     0,   228,    28,    29,     0,     0,    31,     0,     0,
       0,     0,     0,   229,     0,    37,    38,    39,    40,     0,
      42,    43,     0,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,    50,    51,     0,     0,     0,    52,     0,     0,     0,
       0,     0,    53,     0,    54,    55,    56,    57,    58,    59,
       0,    60,    61,    62,    63,     4,     0,     5,     6,     7,
       8,     9,    10,    11,   514,    13,    14,    15,    16,     0,
      17,     0,     0,    18,     0,     0,     0,    19,     0,     0,
      21,    22,    23,    24,     0,    25,   227,     0,   228,    28,
      29,     0,     0,    31,     0,     0,     0,     0,     0,   229,
       0,    37,    38,    39,    40,     0,    42,    43,     0,    44,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,    50,    51,     0,
       0,     0,    52,     0,     0,     0,     0,     0,    53,     0,
      54,    55,    56,    57,    58,    59,     0,    60,    61,    62,
      63,     4,     0,     5,     6,     7,     8,     9,    10,    11,
     550,    13,    14,    15,    16,     0,    17,     0,     0,    18,
       0,     0,     0,    19,     0,     0,    21,    22,    23,    24,
       0,    25,   227,     0,   228,    28,    29,     0,     0,    31,
       0,     0,     0,     0,     0,   229,     0,    37,    38,    39,
      40,     0,    42,    43,     0,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,    50,    51,     0,     0,     0,    52,     0,
       0,     0,     0,     0,    53,     0,    54,    55,    56,    57,
      58,    59,     0,    60,    61,    62,    63,     4,     0,     5,
       6,     7,     8,     9,    10,    11,   605,    13,    14,    15,
      16,     0,    17,     0,     0,    18,     0,     0,     0,    19,
       0,     0,    21,    22,    23,    24,     0,    25,   227,     0,
     228,    28,    29,     0,     0,    31,     0,     0,     0,     0,
       0,   229,     0,    37,    38,    39,    40,     0,    42,    43,
       0,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,    50,
      51,     0,     0,     0,    52,     0,     0,     0,     0,     0,
      53,     0,    54,    55,    56,    57,    58,    59,     0,    60,
      61,    62,    63,     4,     0,     5,     6,     7,     8,     9,
      10,    11,   660,    13,    14,    15,    16,     0,    17,     0,
       0,    18,     0,     0,     0,    19,     0,     0,    21,    22,
      23,    24,     0,    25,   227,     0,   228,    28,    29,     0,
       0,    31,     0,     0,     0,     0,     0,   229,     0,    37,
      38,    39,    40,     0,    42,    43,     0,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,    50,    51,     0,     0,     0,
      52,     0,     0,     0,     0,     0,    53,     0,    54,    55,
      56,    57,    58,    59,     0,    60,    61,    62,    63,     4,
       0,     5,     6,     7,     8,     9,    10,    11,   708,    13,
      14,    15,    16,     0,    17,     0,     0,    18,     0,     0,
       0,    19,     0,     0,    21,    22,    23,    24,     0,    25,
     227,     0,   228,    28,    29,     0,     0,    31,     0,     0,
       0,     0,     0,   229,     0,    37,    38,    39,    40,     0,
      42,    43,     0,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,    50,    51,     0,     0,     0,    52,     0,     0,     0,
       0,     0,    53,     0,    54,    55,    56,    57,    58,    59,
       0,    60,    61,    62,    63,     4,     0,     5,     6,     7,
       8,     9,    10,    11,   709,    13,    14,    15,    16,     0,
      17,     0,     0,    18,     0,     0,     0,    19,     0,     0,
      21,    22,    23,    24,     0,    25,   227,     0,   228,    28,
      29,     0,     0,    31,     0,     0,     0,     0,     0,   229,
       0,    37,    38,    39,    40,     0,    42,    43,     0,    44,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,    50,    51,     0,
       0,     0,    52,     0,     0,     0,     0,     0,    53,     0,
      54,    55,    56,    57,    58,    59,     0,    60,    61,    62,
      63,     4,     0,     5,     6,     7,     8,     9,    10,    11,
       0,    13,    14,    15,    16,     0,    17,     0,     0,    18,
       0,     0,     0,    19,     0,     0,    21,    22,    23,    24,
       0,    25,   227,     0,   228,    28,    29,     0,     0,    31,
       0,     0,     0,     0,     0,   229,     0,    37,    38,    39,
      40,     0,    42,    43,     0,    44,     0,    45,     0,    46,
       0,    47,   719,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,    50,    51,     0,     0,     0,    52,     0,
       0,     0,     0,     0,    53,     0,    54,    55,    56,    57,
      58,    59,     0,    60,    61,    62,    63,     4,     0,     5,
       6,     7,     8,     9,    10,    11,   -78,    13,    14,    15,
      16,     0,    17,     0,     0,    18,     0,     0,     0,    19,
       0,     0,    21,    22,    23,    24,     0,    25,   227,     0,
     228,    28,    29,     0,     0,    31,     0,     0,     0,     0,
       0,   229,     0,    37,    38,    39,    40,     0,    42,    43,
       0,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,    50,
      51,     0,     0,     0,    52,     0,     0,     0,     0,     0,
      53,     0,    54,    55,    56,    57,    58,    59,     0,    60,
      61,    62,    63,     4,     0,     5,     6,     7,     8,     9,
      10,    11,   824,    13,    14,    15,    16,     0,    17,     0,
       0,    18,     0,     0,     0,    19,     0,     0,    21,    22,
      23,    24,     0,    25,   227,     0,   228,    28,    29,     0,
       0,    31,     0,     0,     0,     0,     0,   229,     0,    37,
      38,    39,    40,     0,    42,    43,     0,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,    50,    51,     0,     0,     0,
      52,     0,     0,     0,     0,     0,    53,     0,    54,    55,
      56,    57,    58,    59,     0,    60,    61,    62,    63,     4,
       0,     5,     6,     7,     8,     9,    10,    11,   825,    13,
      14,    15,    16,     0,    17,     0,     0,    18,     0,     0,
       0,    19,     0,     0,    21,    22,    23,    24,     0,    25,
     227,     0,   228,    28,    29,     0,     0,    31,     0,     0,
       0,     0,     0,   229,     0,    37,    38,    39,    40,     0,
      42,    43,     0,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,    50,    51,     0,     0,     0,    52,     0,     0,     0,
       0,     0,    53,     0,    54,    55,    56,    57,    58,    59,
       0,    60,    61,    62,    63,     4,     0,     5,     6,     7,
       8,     9,    10,    11,   826,    13,    14,    15,    16,     0,
      17,     0,     0,    18,     0,     0,     0,    19,     0,     0,
      21,    22,    23,    24,     0,    25,   227,     0,   228,    28,
      29,     0,     0,    31,     0,     0,     0,     0,     0,   229,
       0,    37,    38,    39,    40,     0,    42,    43,     0,    44,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,    50,    51,     0,
       0,     0,    52,     0,     0,     0,     0,     0,    53,     0,
      54,    55,    56,    57,    58,    59,     0,    60,    61,    62,
      63,     4,     0,     5,     6,     7,     8,     9,    10,    11,
    -154,    13,    14,    15,    16,     0,    17,     0,     0,    18,
       0,     0,     0,    19,     0,     0,    21,    22,    23,    24,
       0,    25,   227,     0,   228,    28,    29,     0,     0,    31,
       0,     0,     0,     0,     0,   229,     0,    37,    38,    39,
      40,     0,    42,    43,     0,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,    50,    51,     0,     0,     0,    52,     0,
       0,     0,     0,     0,    53,     0,    54,    55,    56,    57,
      58,    59,     0,    60,    61,    62,    63,     4,     0,     5,
       6,     7,     8,     9,    10,    11,   846,    13,    14,    15,
      16,     0,    17,     0,     0,    18,     0,     0,     0,    19,
       0,     0,    21,    22,    23,    24,     0,    25,   227,     0,
     228,    28,    29,     0,     0,    31,     0,     0,     0,     0,
       0,   229,     0,    37,    38,    39,    40,     0,    42,    43,
       0,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,    50,
      51,     0,     0,     0,    52,     0,     0,     0,     0,     0,
      53,     0,    54,    55,    56,    57,    58,    59,     0,    60,
      61,    62,    63,     4,     0,     5,     6,     7,     8,     9,
      10,    11,     0,    13,    14,    15,    16,     0,    17,     0,
       0,    18,     0,     0,     0,    19,     0,     0,    21,    22,
      23,    24,     0,    25,   227,     0,   228,    28,    29,     0,
       0,    31,     0,     0,     0,     0,     0,   229,     0,    37,
      38,    39,    40,     0,    42,    43,     0,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   156,     0,   157,     6,     7,
     129,     9,    10,    11,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,    50,    51,     0,     0,     0,
      52,    22,    23,     0,     0,     0,    53,     0,    54,    55,
      56,    57,    58,    59,     0,    60,    61,    62,    63,   130,
       0,    37,     0,    39,     0,     0,    42,    43,     0,     0,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   202,     0,   203,
       6,     7,   129,     9,    10,    11,     0,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,    51,     0,
       0,     0,    52,    22,    23,     0,     0,     0,     0,     0,
      54,    55,    56,    57,    58,    59,     0,    60,    61,    62,
      63,   130,     0,    37,     0,    39,     0,     0,    42,    43,
       0,     0,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   209,
       0,   210,     6,     7,   129,     9,    10,    11,     0,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
      51,     0,     0,     0,    52,    22,    23,     0,     0,     0,
       0,     0,    54,    55,    56,    57,    58,    59,     0,    60,
      61,    62,    63,   130,     0,    37,     0,    39,     0,     0,
      42,    43,     0,     0,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   128,     0,     0,     6,     7,   129,     9,    10,    11,
       0,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,    51,     0,     0,     0,    52,    22,    23,     0,
       0,     0,     0,     0,    54,    55,    56,    57,    58,    59,
       0,    60,    61,    62,    63,   130,     0,    37,     0,    39,
       0,     0,    42,    43,     0,     0,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   137,     0,     0,     6,     7,   129,     9,
      10,    11,     0,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,    51,     0,     0,     0,    52,    22,
      23,     0,     0,     0,     0,     0,    54,    55,    56,    57,
      58,    59,     0,    60,    61,    62,    63,   130,     0,    37,
       0,    39,     0,     0,    42,    43,     0,     0,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   142,     0,     0,     6,     7,
     129,     9,    10,    11,     0,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,    51,     0,     0,     0,
      52,    22,    23,     0,     0,     0,     0,     0,    54,    55,
      56,    57,    58,    59,     0,    60,    61,    62,    63,   130,
       0,    37,     0,    39,     0,     0,    42,    43,     0,     0,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   145,     0,     0,
       6,     7,   129,     9,    10,    11,     0,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,    51,     0,
       0,     0,    52,    22,    23,     0,     0,     0,     0,     0,
      54,    55,    56,    57,    58,    59,     0,    60,    61,    62,
      63,   130,     0,    37,     0,    39,     0,     0,    42,    43,
       0,     0,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   150,
       0,     0,     6,     7,   129,     9,    10,    11,     0,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
      51,     0,     0,     0,    52,    22,    23,     0,     0,     0,
       0,     0,    54,    55,    56,    57,    58,    59,     0,    60,
      61,    62,    63,   130,     0,    37,     0,    39,     0,     0,
      42,    43,     0,     0,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   163,     0,     0,     6,     7,   129,     9,    10,    11,
       0,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,    51,     0,     0,     0,    52,    22,    23,     0,
       0,     0,     0,     0,    54,    55,    56,    57,    58,    59,
       0,    60,    61,    62,    63,   130,     0,    37,     0,    39,
       0,     0,    42,    43,     0,     0,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   184,     0,     0,     6,     7,   129,     9,
      10,    11,     0,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,    51,     0,     0,     0,    52,    22,
      23,     0,     0,     0,     0,     0,    54,    55,    56,    57,
      58,    59,     0,    60,    61,    62,    63,   130,     0,    37,
       0,    39,     0,     0,    42,    43,     0,     0,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   432,     0,     0,     6,     7,
     129,     9,    10,    11,     0,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,    51,     0,     0,     0,
      52,    22,    23,     0,     0,     0,     0,     0,    54,    55,
      56,    57,    58,    59,     0,    60,    61,    62,    63,   130,
       0,    37,     0,    39,     0,     0,    42,    43,     0,     0,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   467,     0,     0,
       6,     7,   129,     9,    10,    11,     0,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,    51,     0,
       0,     0,    52,    22,    23,     0,     0,     0,     0,     0,
      54,    55,    56,    57,    58,    59,     0,    60,    61,    62,
      63,   130,     0,    37,     0,    39,     0,     0,    42,    43,
       0,     0,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   615,
       0,     0,     6,     7,   129,     9,    10,    11,     0,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
      51,     0,     0,     0,    52,    22,    23,     0,     0,     0,
       0,     0,    54,    55,    56,    57,    58,    59,     0,    60,
      61,    62,    63,   130,     0,    37,     0,    39,     0,     0,
      42,    43,     0,     0,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   668,     0,     0,     6,     7,   129,     9,    10,    11,
       0,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,    51,     0,     0,     0,    52,    22,    23,     0,
       0,     0,     0,     0,    54,    55,    56,    57,    58,    59,
       0,    60,    61,    62,    63,   130,     0,    37,     0,    39,
       0,     0,    42,    43,     0,     0,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   672,     0,     0,     6,     7,   129,     9,
      10,    11,     0,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,    51,     0,     0,     0,    52,    22,
      23,     0,     0,     0,     0,     0,    54,    55,    56,    57,
      58,    59,     0,    60,    61,    62,    63,   130,     0,    37,
       0,    39,     0,     0,    42,    43,     0,     0,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   810,     0,     0,     6,     7,
     129,     9,    10,    11,     0,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,    51,     0,     0,     0,
      52,    22,    23,     0,     0,     0,     0,     0,    54,    55,
      56,    57,    58,    59,     0,    60,    61,    62,    63,   130,
       0,    37,     0,    39,     0,     0,    42,    43,     0,     0,
       0,    45,     0,    46,     0,    47,   763,     0,  -102,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,    51,     0,
       0,     0,    52,     0,     0,     0,     0,     0,     0,  -102,
      54,    55,    56,    57,    58,    59,     0,    60,    61,    62,
      63,     0,   246,     0,   247,   248,     0,     0,   561,     0,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,     0,     0,   764,   261,   262,   263,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,     0,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   562,     0,     0,     0,     0,     0,     0,     0,   284,
     285,     0,     0,     0,   246,     0,   247,   248,     0,     0,
       0,     0,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   307,     0,     0,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,     0,     0,     0,     0,     0,     0,     0,
       0,   284,   285,     0,     0,   308,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   246,     0,
     247,   248,     0,     0,     0,     0,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   314,     0,
       0,   261,   262,   263,     0,   264,   265,   266,   267,   268,
     269,   270,   271,   272,     0,     0,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,     0,     0,     0,
       0,     0,     0,     0,     0,   284,   285,     0,     0,   315,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   574,
       0,     0,   246,     0,   247,   248,     0,     0,     0,     0,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,     0,     0,     0,   261,   262,   263,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,     0,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   245,     0,   246,     0,   247,   248,     0,     0,   284,
     285,   249,   250,   251,   252,   253,   254,   255,   256,   257,
     258,   259,   260,     0,     0,   575,   261,   262,   263,     0,
     264,   265,   266,   267,   268,   269,   270,   271,   272,     0,
       0,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   321,     0,     0,   246,     0,   247,   248,     0,
     284,   285,     0,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,     0,     0,     0,   261,   262,
     263,     0,   264,   265,   266,   267,   268,   269,   270,   271,
     272,     0,     0,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   324,     0,   246,     0,   247,   248,
       0,     0,   284,   285,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,     0,     0,     0,   261,
     262,   263,     0,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   328,     0,     0,   246,     0,
     247,   248,     0,   284,   285,     0,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,     0,     0,
       0,   261,   262,   263,     0,   264,   265,   266,   267,   268,
     269,   270,   271,   272,     0,     0,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   334,     0,   246,
       0,   247,   248,     0,     0,   284,   285,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,     0,
       0,     0,   261,   262,   263,     0,   264,   265,   266,   267,
     268,   269,   270,   271,   272,     0,     0,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   367,     0,
       0,   246,     0,   247,   248,     0,   284,   285,     0,   249,
     250,   251,   252,   253,   254,   255,   256,   257,   258,   259,
     260,     0,     0,     0,   261,   262,   263,     0,   264,   265,
     266,   267,   268,   269,   270,   271,   272,     0,     0,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     722,     0,   246,     0,   247,   248,     0,     0,   284,   285,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,     0,     0,     0,   261,   262,   263,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,     0,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   836,     0,     0,   246,     0,   247,   248,     0,   284,
     285,     0,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,     0,     0,     0,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,     0,     0,   246,     0,   247,   248,     0,
       0,   284,   285,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,     0,     0,     0,   261,   262,
     263,     0,   264,   265,   266,   267,   268,   269,   270,   271,
     272,     0,     0,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,     6,     7,   129,     9,    10,    11,
       0,     0,   284,   285,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    22,    23,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   194,   130,     0,    37,     0,    39,
       0,     0,    42,    43,     0,     0,     0,    45,   195,    46,
       0,    47,     0,   196,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     6,     7,   129,
       9,    10,    11,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,    51,     0,     0,     0,    52,     0,
      22,    23,     0,     0,     0,     0,    54,    55,    56,    57,
      58,    59,     0,    60,    61,    62,    63,   194,   130,     0,
      37,     0,    39,     0,     0,    42,    43,     0,     0,     0,
      45,     0,    46,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     6,
       7,   129,     9,    10,    11,     0,    48,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,    22,    23,     0,   417,     0,     0,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
     130,     0,    37,     0,    39,     0,     0,    42,    43,     0,
       0,     0,    45,   186,    46,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     6,     7,   129,     9,    10,    11,     0,    48,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,    22,    23,     0,     0,     0,     0,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   130,     0,    37,     0,    39,     0,     0,    42,
      43,     0,     0,     0,    45,   377,    46,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     6,     7,   129,     9,    10,    11,     0,
      48,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,    22,    23,     0,     0,
       0,     0,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,   130,     0,    37,     0,    39,     0,
       0,    42,    43,     0,     0,   415,    45,     0,    46,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     6,     7,   129,     9,    10,
      11,     0,    48,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,    22,    23,
       0,     0,     0,     0,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,   130,     0,    37,     0,
      39,     0,     0,    42,    43,     0,     0,     0,    45,   504,
      46,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     6,     7,   129,
       9,    10,    11,     0,    48,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
      22,    23,     0,     0,     0,     0,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,   130,     0,
      37,     0,    39,     0,     0,    42,    43,     0,     0,   774,
      45,     0,    46,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     6,
       7,   129,     9,    10,    11,     0,    48,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,    22,    23,   246,     0,   247,   248,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
     130,     0,    37,     0,    39,     0,     0,    42,    43,     0,
       0,     0,    45,     0,    46,     0,    47,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,     0,     0,     0,     0,     0,    48,     0,
       0,   284,   285,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,     0,     0,   380,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   246,     0,   247,   248,     0,     0,   381,     0,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,     0,     0,     0,   261,   262,   263,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,     0,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   380,     0,     0,     0,     0,     0,     0,     0,   284,
     285,     0,     0,     0,   246,   559,   247,   248,     0,     0,
       0,     0,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,     0,     0,     0,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   607,     0,     0,     0,     0,     0,     0,
       0,   284,   285,     0,     0,     0,   246,   608,   247,   248,
       0,     0,     0,     0,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,     0,     0,     0,   261,
     262,   263,     0,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,     0,   376,   246,     0,   247,
     248,     0,     0,   284,   285,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,   259,   260,     0,     0,     0,
     261,   262,   263,     0,   264,   265,   266,   267,   268,   269,
     270,   271,   272,     0,     0,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   283,     0,     0,   246,   503,
     247,   248,     0,     0,   284,   285,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,     0,     0,
       0,   261,   262,   263,     0,   264,   265,   266,   267,   268,
     269,   270,   271,   272,     0,     0,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,     0,     0,   246,
       0,   247,   248,     0,     0,   284,   285,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,     0,
     577,     0,   261,   262,   263,     0,   264,   265,   266,   267,
     268,   269,   270,   271,   272,     0,     0,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,     0,     0,
       0,     0,   246,     0,   247,   248,   284,   285,   609,     0,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,     0,     0,     0,   261,   262,   263,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,     0,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,     0,     0,   246,   666,   247,   248,     0,     0,   284,
     285,   249,   250,   251,   252,   253,   254,   255,   256,   257,
     258,   259,   260,     0,     0,     0,   261,   262,   263,     0,
     264,   265,   266,   267,   268,   269,   270,   271,   272,     0,
       0,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,     0,     0,   246,   788,   247,   248,     0,     0,
     284,   285,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,     0,     0,     0,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,     0,     0,   246,     0,   247,   248,     0,
       0,   284,   285,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,     0,     0,     0,   261,   262,
     263,     0,   264,   265,   266,   267,   268,   269,   270,   271,
     272,     0,     0,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   246,     0,   247,   248,     0,     0,
       0,     0,   284,   285,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   246,     0,   247,   248,     0,     0,     0,
       0,   284,   285,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   262,   263,     0,
     264,   265,   266,   267,   268,   269,   270,   271,   272,     0,
       0,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   246,     0,   247,   248,     0,     0,     0,     0,
     284,   285,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   263,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,     0,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   246,     0,   247,   248,     0,     0,     0,     0,   284,
     285,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   264,   265,
     266,   267,   268,   269,   270,   271,   272,     0,     0,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
       0,     0,     0,     0,     0,     0,     0,     0,   284,   285
};

static const yytype_int16 yycheck[] =
{
       2,    14,    13,   358,    17,   345,    19,    18,    21,   244,
      46,     4,    25,    49,     1,    28,   581,    53,    31,     6,
     364,   365,     1,   588,     3,    38,    39,     0,     4,     6,
       6,     7,    45,    46,     1,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,   391,    61,    62,
       1,     1,    63,    76,     4,     5,    79,     7,     8,     9,
      37,     1,     6,     7,     4,    79,     6,     7,     8,    56,
      72,     3,     1,     1,    76,     1,     4,     5,    92,     7,
       8,     9,     4,     1,     6,     7,     8,     1,     6,     1,
      57,    93,    79,     3,     1,     3,     3,     4,     3,     6,
      79,     3,    52,    53,     3,    98,    57,     1,     1,   122,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    98,    16,    52,    53,    19,     3,   363,    58,
      23,    24,    25,    26,    27,    28,    29,    63,    31,    32,
       6,    34,    35,    36,    58,    63,    39,   149,    98,    46,
      79,    44,    45,    79,    47,    48,    49,    50,    98,    52,
      53,    79,    55,    57,    57,    79,    59,    44,    61,    79,
      98,    79,    77,     1,    56,     3,    98,    79,   533,     4,
       5,   194,     7,     8,     9,    92,     3,    92,     3,   191,
      83,     3,     3,     1,    87,     3,     3,    79,    76,    76,
      93,    94,     4,    79,     6,    98,     3,     3,   773,     1,
     246,   104,    64,   106,   107,   108,   109,   110,   111,     3,
     113,   114,   115,   116,     3,    33,     3,    52,    53,     3,
       1,     1,   234,   246,   247,   470,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,     1,   271,   272,
       3,     1,    77,     3,    56,    44,     3,    79,    79,     3,
      44,    79,    79,    98,   287,   288,     1,    92,     3,     4,
      92,     6,    79,    79,     3,    56,    56,    79,     4,   231,
       6,   304,   303,    33,     1,   237,     3,   239,     1,     3,
      79,    44,    76,     6,   317,    79,   319,   318,    79,    79,
       3,    27,    56,     1,     1,     3,     3,     1,    57,    44,
      59,    60,     6,     1,    57,    44,    59,    60,     6,     4,
      33,     6,     1,     1,     3,    79,    79,    44,     6,    79,
      77,     3,     3,    77,    57,    33,    59,    60,    57,    10,
      59,    60,    27,     3,    79,    76,     3,    44,   371,    98,
      99,   100,   101,   102,   103,   104,   105,   380,   381,    57,
     103,   104,   105,   386,   109,   114,   115,     3,    44,    57,
       3,   114,   115,   738,    45,    60,   741,   100,   101,   102,
     103,   104,   105,     1,    44,     3,   131,    44,    64,   401,
       3,   114,   115,   138,   417,   114,   115,    58,   143,   411,
       3,   146,     3,     3,     3,     6,   151,     6,    44,    10,
     581,    10,     3,   158,     1,    33,     3,   588,     4,   164,
       6,   373,     3,   446,   447,   448,   449,   450,   451,   452,
     453,   454,   455,   456,   784,   606,    37,     1,    37,     3,
     185,    44,   187,     3,    45,    46,    45,    46,   193,     6,
       7,    57,   197,    59,    60,    54,   201,    44,   410,   204,
     814,   206,   207,   208,     1,   815,     3,   212,   213,   214,
     215,   216,   217,     1,     1,     3,     3,   222,   223,     1,
      44,     3,    10,     3,     1,     3,    76,     4,     5,     6,
       7,     8,     9,     3,   517,     1,    24,    25,   104,   105,
       6,     7,     3,     3,     1,     3,     3,    44,   114,   115,
      27,    28,     3,     3,   560,   527,    44,    44,    15,     4,
       3,     6,    44,     3,    44,     1,    44,     3,    45,   575,
      47,     3,    49,     1,    10,    52,    53,   560,     6,   562,
      57,    58,    59,    44,    61,     1,   498,   499,    24,    25,
       6,     3,   575,   576,   577,   300,     3,     3,     1,     6,
      76,   513,    79,    10,     1,     1,    83,     3,    44,     6,
       6,     3,     3,     1,     1,     1,    93,    94,     6,     6,
       6,    98,   605,     3,   607,     3,   609,     3,     1,   106,
     107,   108,   109,   110,   111,   547,   113,   114,   115,   116,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     622,     3,    57,   625,    59,    60,   628,     6,     6,     3,
       3,     6,     4,     3,     3,     6,     3,    76,     6,     3,
       6,     6,   653,   378,     3,    58,     3,     3,    63,    33,
       3,     6,     3,     3,     3,   691,    10,     3,     6,     6,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    10,     3,    10,     3,    10,   618,   619,   691,   114,
     115,     3,    10,   418,     3,   420,   421,   422,   423,   424,
     425,   426,   427,   428,   429,   430,   431,    30,   433,   434,
     435,   436,   437,   438,   439,   440,   441,     3,   443,   444,
       3,    56,     6,    57,     6,     6,    79,     6,     3,     3,
       3,     3,   457,   458,     3,     3,     3,     3,   463,     3,
     465,     3,     3,   468,     3,    78,    57,     3,    59,    60,
      44,    56,   744,    56,     3,   756,     3,    79,     6,     3,
       6,   764,     3,     3,   767,     3,     3,     3,     3,     3,
       3,    56,    76,     3,     3,   500,     3,     3,     3,     3,
     505,   506,     3,   508,     3,     3,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   727,   725,   404,   730,   527,
     750,   733,   794,   114,   115,   797,   741,   545,   755,   360,
     802,   743,   804,   693,   512,   472,   588,   588,   823,    31,
      -1,    -1,    -1,   478,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   183,    -1,   558,    -1,    -1,    -1,    -1,   563,   564,
     565,   566,   567,   568,   569,   570,   571,   572,   573,    -1,
      -1,    -1,    -1,    -1,    57,    -1,    59,    60,    -1,    -1,
      -1,   793,    -1,    -1,   796,    -1,    -1,    -1,    -1,   801,
      -1,   803,    -1,    76,    -1,    -1,    79,    80,    81,    82,
      -1,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,   616,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,   837,    -1,    -1,    -1,    -1,
      -1,   114,   115,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     0,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    -1,    16,    -1,   669,    19,   671,    -1,   673,    23,
      24,    -1,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    -1,    38,    39,    40,    41,    42,    43,
      -1,    45,    -1,    47,    48,    49,    50,    51,    52,    53,
      -1,    55,    -1,    57,    -1,    59,   711,    61,    -1,    -1,
      -1,    -1,   717,   718,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,
     104,    -1,   106,   107,   108,   109,   110,   111,    -1,   113,
     114,   115,   116,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    -1,    16,    -1,
      -1,    19,    20,    21,    22,    23,    -1,    -1,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    -1,
      -1,    39,    -1,    -1,    -1,    -1,   811,    45,   813,    47,
      48,    49,    50,    -1,    52,    53,    -1,    55,    -1,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    87,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,
     108,   109,   110,   111,    -1,   113,   114,   115,   116,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,
      -1,    23,    24,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,
      -1,    -1,    44,    45,    -1,    47,    48,    49,    50,    -1,
      52,    53,    -1,    55,    -1,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    83,    -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,
      -1,    -1,   104,    -1,   106,   107,   108,   109,   110,   111,
      -1,   113,   114,   115,   116,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    -1,
      16,    -1,    -1,    19,    -1,    -1,    -1,    23,    24,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    44,    45,
      -1,    47,    48,    49,    50,    -1,    52,    53,    -1,    55,
      -1,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,
      -1,    87,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,
     106,   107,   108,   109,   110,   111,    -1,   113,   114,   115,
     116,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    -1,    16,    -1,    -1,    19,
      -1,    -1,    -1,    23,    24,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    44,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    -1,    55,    -1,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    83,    -1,    -1,    -1,    87,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,
     110,   111,    -1,   113,   114,   115,   116,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    -1,    16,    17,    18,    19,    -1,    -1,    -1,    23,
      -1,    -1,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,
      -1,    45,    -1,    47,    48,    49,    50,    -1,    52,    53,
      -1,    55,    -1,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,
     104,    -1,   106,   107,   108,   109,   110,   111,    -1,   113,
     114,   115,   116,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    -1,    16,    -1,
      -1,    19,    -1,    -1,    -1,    23,    -1,    -1,    26,    27,
      28,    29,    30,    31,    32,    -1,    34,    35,    36,    -1,
      -1,    39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,
      48,    49,    50,    -1,    52,    53,    -1,    55,    -1,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    87,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,
     108,   109,   110,   111,    -1,   113,   114,   115,   116,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,
      -1,    23,    -1,    -1,    26,    27,    28,    29,    30,    31,
      32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,
      -1,    -1,    -1,    45,    -1,    47,    48,    49,    50,    -1,
      52,    53,    -1,    55,    -1,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    83,    -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,
      -1,    -1,   104,    -1,   106,   107,   108,   109,   110,   111,
      -1,   113,   114,   115,   116,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    -1,
      16,    -1,    -1,    19,    -1,    -1,    -1,    23,    -1,    -1,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,    45,
      -1,    47,    48,    49,    50,    -1,    52,    53,    -1,    55,
      -1,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,
      -1,    87,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,
     106,   107,   108,   109,   110,   111,    -1,   113,   114,   115,
     116,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    -1,    16,    -1,    -1,    19,
      -1,    -1,    -1,    23,    -1,    -1,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    -1,    55,    -1,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    83,    -1,    -1,    -1,    87,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,
     110,   111,    -1,   113,   114,   115,   116,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,    23,
      -1,    -1,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,
      -1,    45,    -1,    47,    48,    49,    50,    -1,    52,    53,
      -1,    55,    -1,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,
     104,    -1,   106,   107,   108,   109,   110,   111,    -1,   113,
     114,   115,   116,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    -1,    16,    -1,
      -1,    19,    -1,    -1,    -1,    23,    -1,    -1,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    -1,
      -1,    39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,
      48,    49,    50,    -1,    52,    53,    -1,    55,    -1,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    87,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,
     108,   109,   110,   111,    -1,   113,   114,   115,   116,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,
      -1,    23,    -1,    -1,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,
      -1,    -1,    -1,    45,    -1,    47,    48,    49,    50,    -1,
      52,    53,    -1,    55,    -1,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    83,    -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,
      -1,    -1,   104,    -1,   106,   107,   108,   109,   110,   111,
      -1,   113,   114,   115,   116,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    -1,
      16,    -1,    -1,    19,    -1,    -1,    -1,    23,    -1,    -1,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,    45,
      -1,    47,    48,    49,    50,    -1,    52,    53,    -1,    55,
      -1,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,
      -1,    87,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,
     106,   107,   108,   109,   110,   111,    -1,   113,   114,   115,
     116,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      -1,    11,    12,    13,    14,    -1,    16,    -1,    -1,    19,
      -1,    -1,    -1,    23,    -1,    -1,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    -1,    55,    -1,    57,    -1,    59,
      -1,    61,    62,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    83,    -1,    -1,    -1,    87,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,
     110,   111,    -1,   113,   114,   115,   116,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,    23,
      -1,    -1,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,
      -1,    45,    -1,    47,    48,    49,    50,    -1,    52,    53,
      -1,    55,    -1,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,
     104,    -1,   106,   107,   108,   109,   110,   111,    -1,   113,
     114,   115,   116,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    -1,    16,    -1,
      -1,    19,    -1,    -1,    -1,    23,    -1,    -1,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    -1,
      -1,    39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,
      48,    49,    50,    -1,    52,    53,    -1,    55,    -1,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    87,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,
     108,   109,   110,   111,    -1,   113,   114,   115,   116,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,
      -1,    23,    -1,    -1,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,
      -1,    -1,    -1,    45,    -1,    47,    48,    49,    50,    -1,
      52,    53,    -1,    55,    -1,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    83,    -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,
      -1,    -1,   104,    -1,   106,   107,   108,   109,   110,   111,
      -1,   113,   114,   115,   116,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    -1,
      16,    -1,    -1,    19,    -1,    -1,    -1,    23,    -1,    -1,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,    45,
      -1,    47,    48,    49,    50,    -1,    52,    53,    -1,    55,
      -1,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,
      -1,    87,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,
     106,   107,   108,   109,   110,   111,    -1,   113,   114,   115,
     116,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    -1,    16,    -1,    -1,    19,
      -1,    -1,    -1,    23,    -1,    -1,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    -1,    55,    -1,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    83,    -1,    -1,    -1,    87,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,
     110,   111,    -1,   113,   114,   115,   116,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,    23,
      -1,    -1,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,
      -1,    45,    -1,    47,    48,    49,    50,    -1,    52,    53,
      -1,    55,    -1,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,
     104,    -1,   106,   107,   108,   109,   110,   111,    -1,   113,
     114,   115,   116,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    -1,    11,    12,    13,    14,    -1,    16,    -1,
      -1,    19,    -1,    -1,    -1,    23,    -1,    -1,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    -1,
      -1,    39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,
      48,    49,    50,    -1,    52,    53,    -1,    55,    -1,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    -1,    83,    -1,    -1,    -1,    87,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    27,    28,    -1,    -1,    -1,   104,    -1,   106,   107,
     108,   109,   110,   111,    -1,   113,   114,   115,   116,    45,
      -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,
      -1,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    -1,    83,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,    -1,
     106,   107,   108,   109,   110,   111,    -1,   113,   114,   115,
     116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,
      -1,    -1,    -1,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    -1,    83,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,
      -1,    -1,   106,   107,   108,   109,   110,   111,    -1,   113,
     114,   115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,
      52,    53,    -1,    -1,    -1,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,     9,
      -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,
      -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,   111,
      -1,   113,   114,   115,   116,    45,    -1,    47,    -1,    49,
      -1,    -1,    52,    53,    -1,    -1,    -1,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,     9,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,
      28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,
     110,   111,    -1,   113,   114,   115,   116,    45,    -1,    47,
      -1,    49,    -1,    -1,    52,    53,    -1,    -1,    -1,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,     9,    -1,    83,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,
     108,   109,   110,   111,    -1,   113,   114,   115,   116,    45,
      -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,
      -1,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,     9,    -1,    83,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,    -1,
     106,   107,   108,   109,   110,   111,    -1,   113,   114,   115,
     116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,
      -1,    -1,    -1,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,     9,    -1,    83,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,
      -1,    -1,   106,   107,   108,   109,   110,   111,    -1,   113,
     114,   115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,
      52,    53,    -1,    -1,    -1,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,     9,
      -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,
      -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,   111,
      -1,   113,   114,   115,   116,    45,    -1,    47,    -1,    49,
      -1,    -1,    52,    53,    -1,    -1,    -1,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,     9,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,
      28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,
     110,   111,    -1,   113,   114,   115,   116,    45,    -1,    47,
      -1,    49,    -1,    -1,    52,    53,    -1,    -1,    -1,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,     9,    -1,    83,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,
     108,   109,   110,   111,    -1,   113,   114,   115,   116,    45,
      -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,
      -1,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,     9,    -1,    83,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,    -1,
     106,   107,   108,   109,   110,   111,    -1,   113,   114,   115,
     116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,
      -1,    -1,    -1,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,     9,    -1,    83,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,
      -1,    -1,   106,   107,   108,   109,   110,   111,    -1,   113,
     114,   115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,
      52,    53,    -1,    -1,    -1,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,     9,
      -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,
      -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,   111,
      -1,   113,   114,   115,   116,    45,    -1,    47,    -1,    49,
      -1,    -1,    52,    53,    -1,    -1,    -1,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,     9,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,
      28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,
     110,   111,    -1,   113,   114,   115,   116,    45,    -1,    47,
      -1,    49,    -1,    -1,    52,    53,    -1,    -1,    -1,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,     9,    -1,    83,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,
     108,   109,   110,   111,    -1,   113,   114,   115,   116,    45,
      -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,
      -1,    57,    -1,    59,    -1,    61,     1,    -1,     3,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,    -1,    44,
     106,   107,   108,   109,   110,   111,    -1,   113,   114,   115,
     116,    -1,    57,    -1,    59,    60,    -1,    -1,     1,    -1,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    -1,    -1,    79,    80,    81,    82,    -1,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    44,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   114,
     115,    -1,    -1,    -1,    57,    -1,    59,    60,    -1,    -1,
      -1,    -1,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,     3,    -1,    -1,    80,    81,    82,
      -1,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   114,   115,    -1,    -1,    44,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    57,    -1,
      59,    60,    -1,    -1,    -1,    -1,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,     3,    -1,
      -1,    80,    81,    82,    -1,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   114,   115,    -1,    -1,    44,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,
      -1,    -1,    57,    -1,    59,    60,    -1,    -1,    -1,    -1,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    -1,    -1,    -1,    80,    81,    82,    -1,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,     3,    -1,    57,    -1,    59,    60,    -1,    -1,   114,
     115,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    -1,    -1,    79,    80,    81,    82,    -1,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,     3,    -1,    -1,    57,    -1,    59,    60,    -1,
     114,   115,    -1,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    -1,    -1,    -1,    80,    81,
      82,    -1,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,     3,    -1,    57,    -1,    59,    60,
      -1,    -1,   114,   115,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    -1,    -1,    -1,    80,
      81,    82,    -1,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,     3,    -1,    -1,    57,    -1,
      59,    60,    -1,   114,   115,    -1,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    -1,    -1,
      -1,    80,    81,    82,    -1,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,     3,    -1,    57,
      -1,    59,    60,    -1,    -1,   114,   115,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    -1,
      -1,    -1,    80,    81,    82,    -1,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,     3,    -1,
      -1,    57,    -1,    59,    60,    -1,   114,   115,    -1,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    -1,    -1,    -1,    80,    81,    82,    -1,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
       3,    -1,    57,    -1,    59,    60,    -1,    -1,   114,   115,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    -1,    -1,    -1,    80,    81,    82,    -1,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,     3,    -1,    -1,    57,    -1,    59,    60,    -1,   114,
     115,    -1,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    -1,    -1,    -1,    80,    81,    82,
      -1,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,    57,    -1,    59,    60,    -1,
      -1,   114,   115,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    -1,    -1,    -1,    80,    81,
      82,    -1,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,     4,     5,     6,     7,     8,     9,
      -1,    -1,   114,   115,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    27,    28,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    44,    45,    -1,    47,    -1,    49,
      -1,    -1,    52,    53,    -1,    -1,    -1,    57,    58,    59,
      -1,    61,    -1,    63,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,
       7,     8,     9,    83,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      27,    28,    -1,    -1,    -1,    -1,   106,   107,   108,   109,
     110,   111,    -1,   113,   114,   115,   116,    44,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    -1,
      57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,
       5,     6,     7,     8,     9,    -1,    83,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    27,    28,    -1,   102,    -1,    -1,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    -1,    57,    58,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     4,     5,     6,     7,     8,     9,    -1,    83,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    -1,    57,    58,    59,    -1,    61,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     4,     5,     6,     7,     8,     9,    -1,
      83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,
      -1,    -1,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    57,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,
       9,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,
      -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    -1,    57,    58,
      59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,
       7,     8,     9,    -1,    83,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,
       5,     6,     7,     8,     9,    -1,    83,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    27,    28,    57,    -1,    59,    60,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    -1,    57,    -1,    59,    -1,    61,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,    -1,    -1,    -1,    83,    -1,
      -1,   114,   115,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,    44,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    57,    -1,    59,    60,    -1,    -1,    63,    -1,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    -1,    -1,    -1,    80,    81,    82,    -1,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    44,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   114,
     115,    -1,    -1,    -1,    57,    58,    59,    60,    -1,    -1,
      -1,    -1,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    -1,    -1,    -1,    80,    81,    82,
      -1,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    44,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   114,   115,    -1,    -1,    -1,    57,    58,    59,    60,
      -1,    -1,    -1,    -1,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    -1,    -1,    -1,    80,
      81,    82,    -1,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    56,    57,    -1,    59,
      60,    -1,    -1,   114,   115,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    -1,    -1,    -1,
      80,    81,    82,    -1,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,    -1,    57,    58,
      59,    60,    -1,    -1,   114,   115,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    -1,    -1,
      -1,    80,    81,    82,    -1,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    -1,    -1,    57,
      -1,    59,    60,    -1,    -1,   114,   115,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    -1,
      78,    -1,    80,    81,    82,    -1,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
      -1,    -1,    57,    -1,    59,    60,   114,   115,    63,    -1,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    -1,    -1,    -1,    80,    81,    82,    -1,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    -1,    57,    58,    59,    60,    -1,    -1,   114,
     115,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    -1,    -1,    -1,    80,    81,    82,    -1,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,    57,    58,    59,    60,    -1,    -1,
     114,   115,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    -1,    -1,    -1,    80,    81,    82,
      -1,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,    57,    -1,    59,    60,    -1,
      -1,   114,   115,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    -1,    -1,    -1,    80,    81,
      82,    -1,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    57,    -1,    59,    60,    -1,    -1,
      -1,    -1,   114,   115,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    81,    82,
      -1,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    57,    -1,    59,    60,    -1,    -1,    -1,
      -1,   114,   115,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    82,    -1,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    57,    -1,    59,    60,    -1,    -1,    -1,    -1,
     114,   115,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    57,    -1,    59,    60,    -1,    -1,    -1,    -1,   114,
     115,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   114,   115
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   118,   119,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    16,    19,    23,
      24,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      38,    39,    40,    41,    42,    43,    45,    47,    48,    49,
      50,    51,    52,    53,    55,    57,    59,    61,    83,    87,
      93,    94,    98,   104,   106,   107,   108,   109,   110,   111,
     113,   114,   115,   116,   120,   121,   123,   124,   125,   127,
     128,   130,   131,   132,   135,   137,   138,   146,   147,   148,
     149,   155,   156,   157,   164,   166,   179,   181,   190,   192,
     199,   200,   201,   203,   205,   213,   214,   215,   217,   218,
     220,   223,   241,   246,   251,   255,   256,   258,   259,   261,
     262,   263,   265,   267,   271,   273,   274,   275,   276,   277,
       3,    44,    64,     3,     1,     6,   126,   258,     1,     6,
      45,   261,     1,     3,     1,     3,    15,     1,   261,     1,
     258,   280,     1,   261,     1,     1,   261,     1,     3,    44,
       1,   261,     1,     6,     1,     6,     1,     3,   261,   252,
       1,     6,     7,     1,   261,   263,     1,     6,     1,     3,
       6,   216,     1,     6,    33,   219,     1,     6,   221,   222,
       1,     6,   266,   272,     1,   261,    58,   261,   278,     1,
       3,    44,     6,   261,    44,    58,    63,   261,   277,   281,
     268,   261,     1,     3,   261,   277,   261,   261,   261,     1,
       3,   277,   261,   261,   261,   261,   261,   261,     4,     6,
      27,    60,   261,   261,     4,   258,   129,    32,    34,    45,
     124,   136,   124,     3,    44,   165,   180,   191,    46,   208,
     211,   212,   124,     1,    57,     3,    57,    59,    60,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    80,    81,    82,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   114,   115,   262,    76,    79,     1,
       4,     5,     7,     8,     9,    52,    53,    98,   122,   257,
     261,     3,     3,    79,    76,     3,    44,     3,    44,     3,
       3,     3,     3,    44,     3,    44,     3,    76,    79,    92,
       3,     3,     3,     3,     3,     3,   124,     3,     3,     3,
     224,     3,   247,     3,     3,     1,     6,   253,   254,     3,
       3,     3,     3,     3,     3,    76,     3,     3,    79,     3,
       1,     6,     7,     1,     3,    33,    79,     3,    76,     3,
      79,     3,     1,    57,   269,   269,     3,     3,     1,    58,
      79,   279,     3,   133,   124,   242,    56,    58,   261,    58,
      44,    63,     1,    58,     1,    58,    79,     1,     6,   206,
     207,   270,     3,     3,     3,     3,     4,     6,    27,   145,
     145,   150,   124,   167,   182,   145,     1,     3,    44,   145,
     209,   210,     3,     1,   206,    56,   277,   102,   261,     6,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,     1,   261,   261,   261,   261,   261,   261,   261,
     261,   261,     6,   261,   261,     3,   260,   260,   260,   260,
     260,   260,   260,   260,   260,   260,   260,   261,   261,     3,
       4,     3,   258,   261,   151,   261,   258,     1,   261,     1,
      57,   225,   226,     1,    33,   228,   248,     3,    79,     1,
       1,   257,     6,     3,     3,    77,     3,    77,     3,     6,
       7,     6,     6,     7,   122,   222,     3,   206,   208,   208,
     261,   145,     3,    58,    58,   261,   261,    58,   261,    63,
       1,    63,    79,   208,    10,   124,    17,    18,   139,   141,
     142,   144,    20,    21,    22,   124,   153,   154,   158,   160,
     162,     1,     3,    24,    25,   168,   173,   175,     1,     3,
      24,   173,   183,    30,   193,   194,   195,   196,     3,    44,
      10,   145,   124,   204,     1,    56,     1,    56,   261,    58,
      79,     1,    44,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,     3,    79,    76,    78,     3,     3,
     206,   232,   228,     3,     6,   229,   230,     3,   249,     1,
     254,     3,     3,     6,     6,     3,    77,    92,     3,    77,
      92,     1,    56,   145,   145,    10,   243,    44,    58,    63,
     207,   145,     3,     1,     3,     1,   261,    10,   140,   143,
       1,     3,    44,     1,     3,    44,     1,     3,    44,    10,
     153,     3,     1,     6,     7,     8,   122,   177,   178,     1,
      10,   174,     3,     1,     4,     6,   188,   189,    10,     1,
       3,     4,     6,    92,   197,   198,    10,   195,   145,     3,
      10,    56,   202,     3,    44,   264,    58,   277,     1,   261,
     277,   261,     1,   261,     1,    56,     3,     6,    10,    37,
      45,    46,    54,   200,   218,   233,   234,   236,   237,   238,
       3,    57,   231,    79,     3,    10,   200,   218,   234,   236,
     250,     3,     3,     6,     6,     6,     6,     3,    10,    10,
     134,   261,     3,     6,    10,   218,   244,   261,   261,    62,
       3,     3,     3,     3,   145,   145,     3,   159,   124,     3,
     161,   124,     3,   163,   124,     3,     3,    44,    78,     3,
      44,    79,     3,     3,    44,   176,     3,    44,     3,    44,
      79,     3,     3,   258,     3,    79,    92,     3,     3,    44,
      56,    56,     3,     1,    79,   152,   227,    76,     3,     3,
       6,     6,    37,   239,    56,   277,   230,     3,     3,     3,
       3,     3,     3,     3,    76,    79,   245,     3,    58,   139,
     145,   145,   145,   171,   172,   122,   169,   170,   178,   145,
     124,   186,   187,   184,   185,   189,     3,   198,   258,     3,
       1,   261,    56,   261,   235,    76,     3,     3,     3,    10,
     200,   240,    56,   257,    10,    10,    10,   145,   124,   145,
     124,   145,   124,   145,   124,     3,     3,   208,   257,     3,
     245,     3,     3,     3,   145,     3,    10,     3
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (&yylval, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval)
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}

/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */





/*-------------------------.
| yyparse or yypush_parse.  |
`-------------------------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

    /* Number of syntax errors so far.  */
    int yynerrs;

    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 6:

/* Line 1455 of yacc.c  */
#line 208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); }
    break;

  case 7:

/* Line 1455 of yacc.c  */
#line 209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); }
    break;

  case 8:

/* Line 1455 of yacc.c  */
#line 213 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat)=0; }
    break;

  case 10:

/* Line 1455 of yacc.c  */
#line 216 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 11:

/* Line 1455 of yacc.c  */
#line 221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 12:

/* Line 1455 of yacc.c  */
#line 226 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 13:

/* Line 1455 of yacc.c  */
#line 231 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 14:

/* Line 1455 of yacc.c  */
#line 236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 19:

/* Line 1455 of yacc.c  */
#line 247 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); }
    break;

  case 20:

/* Line 1455 of yacc.c  */
#line 252 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), false );
      }
    break;

  case 21:

/* Line 1455 of yacc.c  */
#line 258 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), true );
      }
    break;

  case 22:

/* Line 1455 of yacc.c  */
#line 264 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
      }
    break;

  case 23:

/* Line 1455 of yacc.c  */
#line 270 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); }
    break;

  case 24:

/* Line 1455 of yacc.c  */
#line 271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_stat)=0;}
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; }
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; }
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; }
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; }
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;}
    break;

  case 30:

/* Line 1455 of yacc.c  */
#line 281 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (! COMPILER->isInteractive()) && ((!(yyvsp[(1) - (2)].fal_val)->isExpr()) || (!(yyvsp[(1) - (2)].fal_val)->asExpr()->isStandAlone()) ) )
         {
            Falcon::StmtFunction *func = COMPILER->getFunctionContext();
            if ( func == 0 || ! func->isLambda() )
               COMPILER->raiseError(Falcon::e_noeffect );
         }

         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) );
      }
    break;

  case 31:

/* Line 1455 of yacc.c  */
#line 292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
         COMPILER->defineVal( first );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
      }
    break;

  case 32:

/* Line 1455 of yacc.c  */
#line 298 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (6)].fal_adecl)->size() != (yyvsp[(5) - (6)].fal_adecl)->size() + 1 )
         {
            COMPILER->raiseError(Falcon::e_unpack_size );
         }
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (6)].fal_adecl) );

         COMPILER->defineVal( first );
         (yyvsp[(5) - (6)].fal_adecl)->pushFront( (yyvsp[(3) - (6)].fal_val) );
         Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, second ) ) );
      }
    break;

  case 50:

/* Line 1455 of yacc.c  */
#line 332 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_def );
   }
    break;

  case 51:

/* Line 1455 of yacc.c  */
#line 335 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr( CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
   }
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 341 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ) ) );
   }
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 350 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; }
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 352 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); }
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 356 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 56:

/* Line 1455 of yacc.c  */
#line 363 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 370 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 378 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 379 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; }
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 383 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 384 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 62:

/* Line 1455 of yacc.c  */
#line 388 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 63:

/* Line 1455 of yacc.c  */
#line 395 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = static_cast<Falcon::StmtLoop* >(COMPILER->getContext());
         w->setCondition((yyvsp[(6) - (7)].fal_val));
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 64:

/* Line 1455 of yacc.c  */
#line 403 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 65:

/* Line 1455 of yacc.c  */
#line 409 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_loop );
      (yyval.fal_stat) = 0;
   }
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 416 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 67:

/* Line 1455 of yacc.c  */
#line 417 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(1) - (1)].fal_val); }
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 421 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      }
    break;

  case 69:

/* Line 1455 of yacc.c  */
#line 429 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 70:

/* Line 1455 of yacc.c  */
#line 436 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      }
    break;

  case 71:

/* Line 1455 of yacc.c  */
#line 446 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 72:

/* Line 1455 of yacc.c  */
#line 447 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; }
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 77:

/* Line 1455 of yacc.c  */
#line 459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      }
    break;

  case 80:

/* Line 1455 of yacc.c  */
#line 469 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); }
    break;

  case 81:

/* Line 1455 of yacc.c  */
#line 474 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      }
    break;

  case 83:

/* Line 1455 of yacc.c  */
#line 486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 84:

/* Line 1455 of yacc.c  */
#line 487 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; }
    break;

  case 86:

/* Line 1455 of yacc.c  */
#line 492 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   }
    break;

  case 87:

/* Line 1455 of yacc.c  */
#line 499 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      }
    break;

  case 88:

/* Line 1455 of yacc.c  */
#line 508 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      }
    break;

  case 89:

/* Line 1455 of yacc.c  */
#line 516 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      }
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 526 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      }
    break;

  case 91:

/* Line 1455 of yacc.c  */
#line 535 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      }
    break;

  case 92:

/* Line 1455 of yacc.c  */
#line 543 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtForin( LINE, (yyvsp[(2) - (4)].fal_adecl), (yyvsp[(4) - (4)].fal_val) );
      }
    break;

  case 93:

/* Line 1455 of yacc.c  */
#line 548 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) );
         Falcon::ArrayDecl *decl = new Falcon::ArrayDecl();
         decl->pushBack( (yyvsp[(2) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtForin( LINE, decl, (yyvsp[(4) - (4)].fal_val) );  
      }
    break;

  case 94:

/* Line 1455 of yacc.c  */
#line 556 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 95:

/* Line 1455 of yacc.c  */
#line 561 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 96:

/* Line 1455 of yacc.c  */
#line 570 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>((yyvsp[(1) - (3)].fal_stat));
      if( (yyvsp[(3) - (3)].fal_stat) != 0 )
      {
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
         
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
      (yyval.fal_stat) = f;
   }
    break;

  case 97:

/* Line 1455 of yacc.c  */
#line 588 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>((yyvsp[(1) - (2)].fal_stat));
      
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 98:

/* Line 1455 of yacc.c  */
#line 598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 99:

/* Line 1455 of yacc.c  */
#line 609 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::RangeDecl* rd = new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val),
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(3) - (4)].fal_val))), (yyvsp[(4) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( rd );
      }
    break;

  case 100:

/* Line 1455 of yacc.c  */
#line 615 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val), 0 ) );
      }
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 619 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (3)].fal_val), 0, 0 ) );
      }
    break;

  case 102:

/* Line 1455 of yacc.c  */
#line 625 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 103:

/* Line 1455 of yacc.c  */
#line 626 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 104:

/* Line 1455 of yacc.c  */
#line 627 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 107:

/* Line 1455 of yacc.c  */
#line 636 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      }
    break;

  case 111:

/* Line 1455 of yacc.c  */
#line 650 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getLoop());
         if ( f == 0 || f->type() != Falcon::Statement::t_forin )
         {
            COMPILER->raiseError( Falcon::e_syn_fordot );
            delete (yyvsp[(2) - (3)].fal_val);
            (yyval.fal_stat) = 0;
         }
         else {
            (yyval.fal_stat) = new Falcon::StmtFordot( LINE, (yyvsp[(2) - (3)].fal_val) );
         }
      }
    break;

  case 112:

/* Line 1455 of yacc.c  */
#line 663 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      }
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 671 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 681 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 687 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      }
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 694 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 699 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 119:

/* Line 1455 of yacc.c  */
#line 708 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   }
    break;

  case 120:

/* Line 1455 of yacc.c  */
#line 717 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         COMPILER->pushContextSet( &f->firstBlock() );
		 // Push anyhow an empty item, that is needed for to check again for thio blosk
		 f->firstBlock().push_back( new Falcon::StmtNone( LINE ) );
      }
    break;

  case 121:

/* Line 1455 of yacc.c  */
#line 729 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 731 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 740 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); }
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 744 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->lastBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->lastBlock() );
      }
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 756 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 757 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 766 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); }
    break;

  case 128:

/* Line 1455 of yacc.c  */
#line 770 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 // Apparently you get a segfault without it.
		 // (Note that the formiddle: version below does *not* need it
		 f->middleBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->middleBlock() );
      }
    break;

  case 129:

/* Line 1455 of yacc.c  */
#line 784 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 130:

/* Line 1455 of yacc.c  */
#line 786 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->middleBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 131:

/* Line 1455 of yacc.c  */
#line 795 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); }
    break;

  case 132:

/* Line 1455 of yacc.c  */
#line 799 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 133:

/* Line 1455 of yacc.c  */
#line 807 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 134:

/* Line 1455 of yacc.c  */
#line 816 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 135:

/* Line 1455 of yacc.c  */
#line 818 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 138:

/* Line 1455 of yacc.c  */
#line 827 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); }
    break;

  case 140:

/* Line 1455 of yacc.c  */
#line 833 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 142:

/* Line 1455 of yacc.c  */
#line 843 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 143:

/* Line 1455 of yacc.c  */
#line 851 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 144:

/* Line 1455 of yacc.c  */
#line 855 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 146:

/* Line 1455 of yacc.c  */
#line 867 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 147:

/* Line 1455 of yacc.c  */
#line 877 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 149:

/* Line 1455 of yacc.c  */
#line 886 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();
         if ( ! stmt->defaultBlock().empty() )
         {
            COMPILER->raiseError(Falcon::e_switch_default, "", CURRENT_LINE );
         }
         COMPILER->pushContextSet( &stmt->defaultBlock() );
      }
    break;

  case 153:

/* Line 1455 of yacc.c  */
#line 900 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); }
    break;

  case 155:

/* Line 1455 of yacc.c  */
#line 904 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 158:

/* Line 1455 of yacc.c  */
#line 916 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      }
    break;

  case 159:

/* Line 1455 of yacc.c  */
#line 925 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 160:

/* Line 1455 of yacc.c  */
#line 937 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].stringp) );
         if ( ! stmt->addStringCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 161:

/* Line 1455 of yacc.c  */
#line 948 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (yyvsp[(1) - (3)].integer) ), new Falcon::Value( (yyvsp[(3) - (3)].integer) ) ) );
         if ( ! stmt->addRangeCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 162:

/* Line 1455 of yacc.c  */
#line 959 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 163:

/* Line 1455 of yacc.c  */
#line 979 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 164:

/* Line 1455 of yacc.c  */
#line 987 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 165:

/* Line 1455 of yacc.c  */
#line 996 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 166:

/* Line 1455 of yacc.c  */
#line 998 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 169:

/* Line 1455 of yacc.c  */
#line 1007 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); }
    break;

  case 171:

/* Line 1455 of yacc.c  */
#line 1013 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 173:

/* Line 1455 of yacc.c  */
#line 1023 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 174:

/* Line 1455 of yacc.c  */
#line 1032 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 175:

/* Line 1455 of yacc.c  */
#line 1036 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 177:

/* Line 1455 of yacc.c  */
#line 1048 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 178:

/* Line 1455 of yacc.c  */
#line 1058 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 182:

/* Line 1455 of yacc.c  */
#line 1072 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 183:

/* Line 1455 of yacc.c  */
#line 1084 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 184:

/* Line 1455 of yacc.c  */
#line 1104 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   }
    break;

  case 185:

/* Line 1455 of yacc.c  */
#line 1111 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      }
    break;

  case 186:

/* Line 1455 of yacc.c  */
#line 1121 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
    break;

  case 188:

/* Line 1455 of yacc.c  */
#line 1130 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); }
    break;

  case 194:

/* Line 1455 of yacc.c  */
#line 1150 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }
         // but continue by pushing this new context
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }
    break;

  case 195:

/* Line 1455 of yacc.c  */
#line 1168 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }

         // but continue by pushing this new context
         COMPILER->defineVal( (yyvsp[(3) - (4)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(3) - (4)].fal_val) );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }
    break;

  case 196:

/* Line 1455 of yacc.c  */
#line 1188 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 197:

/* Line 1455 of yacc.c  */
#line 1198 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 198:

/* Line 1455 of yacc.c  */
#line 1209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   }
    break;

  case 201:

/* Line 1455 of yacc.c  */
#line 1222 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );

         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 202:

/* Line 1455 of yacc.c  */
#line 1234 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 203:

/* Line 1455 of yacc.c  */
#line 1256 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 204:

/* Line 1455 of yacc.c  */
#line 1257 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; }
    break;

  case 205:

/* Line 1455 of yacc.c  */
#line 1269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 206:

/* Line 1455 of yacc.c  */
#line 1275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 208:

/* Line 1455 of yacc.c  */
#line 1284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 209:

/* Line 1455 of yacc.c  */
#line 1285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 210:

/* Line 1455 of yacc.c  */
#line 1288 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
    break;

  case 212:

/* Line 1455 of yacc.c  */
#line 1293 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 213:

/* Line 1455 of yacc.c  */
#line 1294 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 214:

/* Line 1455 of yacc.c  */
#line 1301 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // if we are in a class, I have to create the symbol classname.functionname
         Falcon::Statement *parent = COMPILER->getContext();
         Falcon::String *func_name;
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::String complete_name = stmt_cls->symbol()->name() + "." + *(yyvsp[(2) - (2)].stringp);
            func_name = COMPILER->addString( complete_name );
         }
         else if ( parent != 0 && parent->type() == Falcon::Statement::t_state ) 
         {
            Falcon::StmtState *stmt_state = static_cast< Falcon::StmtState *>( parent );
            Falcon::String complete_name =  
                  stmt_state->owner()->symbol()->name() + "." + 
                  * stmt_state->name() + "#" + *(yyvsp[(2) - (2)].stringp);
                  
            func_name = COMPILER->addString( complete_name );
         }
         else
            func_name = (yyvsp[(2) - (2)].stringp);

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( func_name );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( func_name );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         // and eventually add it as a class property
         if ( parent != 0 )
         {
            if( parent->type() == Falcon::Statement::t_class ) 
            {
               Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
               Falcon::ClassDef *cd = stmt_cls->symbol()->getClassDef();
               if ( cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) ) {
                  COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (2)].stringp) );
               }
               else 
               {
                   cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef( sym ) );
                   // is this a setter/getter?
                   if( ( (yyvsp[(2) - (2)].stringp)->find( "__set_" ) == 0 || (yyvsp[(2) - (2)].stringp)->find( "__get_" ) == 0 ) && (yyvsp[(2) - (2)].stringp)->length() > 6 )
                   {
                      Falcon::String *pname = COMPILER->addString( (yyvsp[(2) - (2)].stringp)->subString( 6 ));
                      Falcon::VarDef *pd = cd->getProperty( *pname );
                      if( pd == 0 )
                      {
                        pd = new Falcon::VarDef;
                        cd->addProperty( pname, pd );
                        pd->setReflective( Falcon::e_reflectSetGet, 0xFFFFFFFF );
                      }
                      else if( ! pd->isReflective() )
                      {
                        COMPILER->raiseError(Falcon::e_prop_adef, *pname );
                      }
   
                   }
               }
            }
            else if ( parent->type() == Falcon::Statement::t_state ) 
            {
   
               Falcon::StmtState *stmt_state = static_cast< Falcon::StmtState *>( parent );            
               if( ! stmt_state->addFunction( (yyvsp[(2) - (2)].stringp), sym ) )
               {
                  COMPILER->raiseError(Falcon::e_sm_adef, *(yyvsp[(2) - (2)].stringp) );
               }
               else 
               {
                  stmt_state->state()->addFunction( (yyvsp[(2) - (2)].stringp), sym );
               }
               
               // eventually add a property where to store this thing
               Falcon::ClassDef *cd = stmt_state->owner()->symbol()->getClassDef();
               if ( ! cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) )
                  cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef );
            }
         }
         
         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 218:

/* Line 1455 of yacc.c  */
#line 1411 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym != 0 ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }
         else {
            Falcon::FuncDef *func = COMPILER->getFunction();
            Falcon::Symbol *sym = new Falcon::Symbol( COMPILER->module(), (yyvsp[(1) - (1)].stringp) );
            COMPILER->module()->addSymbol( sym );
            func->addParameter( sym );
         }
      }
    break;

  case 220:

/* Line 1455 of yacc.c  */
#line 1428 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 221:

/* Line 1455 of yacc.c  */
#line 1434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 222:

/* Line 1455 of yacc.c  */
#line 1439 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 223:

/* Line 1455 of yacc.c  */
#line 1445 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 225:

/* Line 1455 of yacc.c  */
#line 1454 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); }
    break;

  case 227:

/* Line 1455 of yacc.c  */
#line 1459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); }
    break;

  case 228:

/* Line 1455 of yacc.c  */
#line 1469 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 229:

/* Line 1455 of yacc.c  */
#line 1472 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; }
    break;

  case 230:

/* Line 1455 of yacc.c  */
#line 1481 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = (yyvsp[(4) - (5)].fal_val); //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *(yyvsp[(2) - (5)].stringp), val, LINE );
         // we don't need the expression anymore
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 231:

/* Line 1455 of yacc.c  */
#line 1491 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      }
    break;

  case 232:

/* Line 1455 of yacc.c  */
#line 1496 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      }
    break;

  case 233:

/* Line 1455 of yacc.c  */
#line 1508 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 234:

/* Line 1455 of yacc.c  */
#line 1517 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      }
    break;

  case 235:

/* Line 1455 of yacc.c  */
#line 1524 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      }
    break;

  case 236:

/* Line 1455 of yacc.c  */
#line 1532 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      }
    break;

  case 237:

/* Line 1455 of yacc.c  */
#line 1537 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      }
    break;

  case 238:

/* Line 1455 of yacc.c  */
#line 1545 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (3)].fal_genericList) );
         (yyval.fal_stat) = 0;
      }
    break;

  case 239:

/* Line 1455 of yacc.c  */
#line 1550 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 240:

/* Line 1455 of yacc.c  */
#line 1555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 241:

/* Line 1455 of yacc.c  */
#line 1560 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( symName, (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), false );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );

         (yyval.fal_stat) = 0;
      }
    break;

  case 242:

/* Line 1455 of yacc.c  */
#line 1580 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( symName, (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), true );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 243:

/* Line 1455 of yacc.c  */
#line 1599 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 244:

/* Line 1455 of yacc.c  */
#line 1604 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 245:

/* Line 1455 of yacc.c  */
#line 1609 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 246:

/* Line 1455 of yacc.c  */
#line 1614 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (4)].fal_genericList)->begin();
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            delete symName;
            li = li->next();
         }
         delete (yyvsp[(2) - (4)].fal_genericList);

         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 247:

/* Line 1455 of yacc.c  */
#line 1628 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 248:

/* Line 1455 of yacc.c  */
#line 1633 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 249:

/* Line 1455 of yacc.c  */
#line 1638 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 250:

/* Line 1455 of yacc.c  */
#line 1643 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 251:

/* Line 1455 of yacc.c  */
#line 1648 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 252:

/* Line 1455 of yacc.c  */
#line 1657 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addAttribute( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val), LINE );
     }
    break;

  case 253:

/* Line 1455 of yacc.c  */
#line 1662 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_attrdecl );
     }
    break;

  case 254:

/* Line 1455 of yacc.c  */
#line 1669 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::List *lst = new Falcon::List;
         lst->pushBack( new Falcon::String( *(yyvsp[(1) - (1)].stringp) ) );
         (yyval.fal_genericList) = lst;
      }
    break;

  case 255:

/* Line 1455 of yacc.c  */
#line 1675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(1) - (3)].fal_genericList)->pushBack( new Falcon::String( *(yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_genericList) = (yyvsp[(1) - (3)].fal_genericList);
      }
    break;

  case 256:

/* Line 1455 of yacc.c  */
#line 1687 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 257:

/* Line 1455 of yacc.c  */
#line 1692 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     }
    break;

  case 260:

/* Line 1455 of yacc.c  */
#line 1705 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 261:

/* Line 1455 of yacc.c  */
#line 1709 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 262:

/* Line 1455 of yacc.c  */
#line 1713 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      }
    break;

  case 263:

/* Line 1455 of yacc.c  */
#line 1726 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );
      }
    break;

  case 264:

/* Line 1455 of yacc.c  */
#line 1758 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls != 0 )
         {
            if ( cls->ctorFunction() == 0  )
            {
               Falcon::ClassDef *cd = cls->symbol()->getClassDef();
               if ( ! cd->inheritance().empty() )
               {
                  COMPILER->buildCtorFor( cls );
                  // COMPILER->addStatement( func ); should be done in buildCtorFor
                  // cls->ctorFunction( func ); idem
               }
            }
            COMPILER->popContext();
            //We didn't pushed a context set
            COMPILER->popFunction();
         }
      }
    break;

  case 266:

/* Line 1455 of yacc.c  */
#line 1792 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
    break;

  case 269:

/* Line 1455 of yacc.c  */
#line 1800 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 270:

/* Line 1455 of yacc.c  */
#line 1801 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 275:

/* Line 1455 of yacc.c  */
#line 1818 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol((yyvsp[(1) - (2)].stringp));
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            cls->addInitExpression(
               new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_inherit,
                     new Falcon::Value( sym ), (yyvsp[(2) - (2)].fal_val) ) ) );
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
            delete (yyvsp[(2) - (2)].fal_val);
         }
      }
    break;

  case 276:

/* Line 1455 of yacc.c  */
#line 1841 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 277:

/* Line 1455 of yacc.c  */
#line 1842 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 278:

/* Line 1455 of yacc.c  */
#line 1844 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = (yyvsp[(2) - (3)].fal_adecl) == 0 ? 0 : new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
   }
    break;

  case 282:

/* Line 1455 of yacc.c  */
#line 1857 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 283:

/* Line 1455 of yacc.c  */
#line 1860 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   }
    break;

  case 284:

/* Line 1455 of yacc.c  */
#line 1880 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtState* ss = static_cast<Falcon::StmtState *>((yyvsp[(1) - (1)].fal_stat));
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );         
         if( ! cls->addState( static_cast<Falcon::StmtState *>((yyvsp[(1) - (1)].fal_stat)) ) )
         {
            const Falcon::String* name = ss->name();
            // we're not using the state we created, afater all.
            delete ss->state();
            delete (yyvsp[(1) - (1)].fal_stat);
            COMPILER->raiseError( Falcon::e_state_adef, *name ); 
         }
         else {
            // ownership passes on to the classdef
            cls->symbol()->getClassDef()->addState( ss->name(), ss->state() );
            cls->symbol()->getClassDef()->addProperty( ss->name(), 
               new Falcon::VarDef( ss->name() ) );
         }
      }
    break;

  case 287:

/* Line 1455 of yacc.c  */
#line 1904 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         if( cls->initGiven() ) {
            COMPILER->raiseError(Falcon::e_init_given );
         }
         else
         {
            cls->initGiven( true );
            Falcon::StmtFunction *func = cls->ctorFunction();
            if ( func == 0 ) {
               func = COMPILER->buildCtorFor( cls );
            }

            // prepare the statement allocation context
            COMPILER->pushContext( func );
            COMPILER->pushFunctionContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      }
    break;

  case 288:

/* Line 1455 of yacc.c  */
#line 1929 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
         COMPILER->popFunctionContext();
      }
    break;

  case 289:

/* Line 1455 of yacc.c  */
#line 1939 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(4) - (5)].fal_val)->genVarDef();

      if ( def != 0 ) {
         Falcon::String prop_name = cls->symbol()->name() + "." + *(yyvsp[(2) - (5)].stringp);
         Falcon::Symbol *sym = COMPILER->addGlobalVar( COMPILER->addString(prop_name), def );
         if( clsdef->hasProperty( *(yyvsp[(2) - (5)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (5)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(2) - (5)].stringp), new Falcon::VarDef( Falcon::VarDef::t_reference, sym) );
      }
      else {
         COMPILER->raiseError(Falcon::e_static_const );
      }
      delete (yyvsp[(4) - (5)].fal_val); // the expression is not needed anymore
      (yyval.fal_stat) = 0; // we don't add any statement
   }
    break;

  case 290:

/* Line 1455 of yacc.c  */
#line 1961 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(3) - (4)].fal_val)->genVarDef();

      if ( def != 0 ) {
         if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), def );
         delete (yyvsp[(3) - (4)].fal_val); // the expression is not needed anymore
         (yyval.fal_stat) = 0; // we don't add any statement
      }
      else {
         // create anyhow a nil property
          if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), new Falcon::VarDef() );
         // but also prepare a statement to be executed by the auto-constructor.
         (yyval.fal_stat) = new Falcon::StmtVarDef( LINE, (yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
   }
    break;

  case 291:

/* Line 1455 of yacc.c  */
#line 1993 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { 
         (yyval.fal_stat) = COMPILER->getContext(); 
         COMPILER->popContext();
      }
    break;

  case 292:

/* Line 1455 of yacc.c  */
#line 2001 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass* cls = 
            static_cast<Falcon::StmtClass*>( COMPILER->getContext() );
            
         COMPILER->pushContext( 
            new Falcon::StmtState( (yyvsp[(2) - (3)].stringp), cls ) ); 
      }
    break;

  case 293:

/* Line 1455 of yacc.c  */
#line 2009 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass* cls = 
            static_cast<Falcon::StmtClass*>( COMPILER->getContext() );
         
         Falcon::StmtState* state = new Falcon::StmtState( COMPILER->addString( "init" ), cls );
         cls->initState( state );
         
         COMPILER->pushContext( state ); 
      }
    break;

  case 297:

/* Line 1455 of yacc.c  */
#line 2030 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 298:

/* Line 1455 of yacc.c  */
#line 2041 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
            sym->setEnum( true );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );

         COMPILER->resetEnum();
      }
    break;

  case 299:

/* Line 1455 of yacc.c  */
#line 2075 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 303:

/* Line 1455 of yacc.c  */
#line 2092 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
    break;

  case 305:

/* Line 1455 of yacc.c  */
#line 2097 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      }
    break;

  case 308:

/* Line 1455 of yacc.c  */
#line 2112 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // we create a special symbol for the class.
         Falcon::String cl_name = "%";
         cl_name += *(yyvsp[(2) - (2)].stringp);
         Falcon::Symbol *clsym = COMPILER->addGlobalSymbol( COMPILER->addString( cl_name ) );
         clsym->setClass( def );

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setInstance( clsym );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), clsym );
         cls->singleton( sym );

         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      }
    break;

  case 309:

/* Line 1455 of yacc.c  */
#line 2152 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( !cd->inheritance().empty() )
            {
               COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 311:

/* Line 1455 of yacc.c  */
#line 2180 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
    break;

  case 315:

/* Line 1455 of yacc.c  */
#line 2192 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 316:

/* Line 1455 of yacc.c  */
#line 2195 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   }
    break;

  case 319:

/* Line 1455 of yacc.c  */
#line 2224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
    break;

  case 320:

/* Line 1455 of yacc.c  */
#line 2229 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // raise an error if we are not in a local context
         if ( ! COMPILER->isLocalContext() )
         {
            COMPILER->raiseError(Falcon::e_global_notin_func, "", LINE );
         }
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
      }
    break;

  case 322:

/* Line 1455 of yacc.c  */
#line 2243 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 323:

/* Line 1455 of yacc.c  */
#line 2248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 325:

/* Line 1455 of yacc.c  */
#line 2254 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 326:

/* Line 1455 of yacc.c  */
#line 2261 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
    break;

  case 327:

/* Line 1455 of yacc.c  */
#line 2276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); }
    break;

  case 328:

/* Line 1455 of yacc.c  */
#line 2277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 329:

/* Line 1455 of yacc.c  */
#line 2278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; }
    break;

  case 330:

/* Line 1455 of yacc.c  */
#line 2286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 331:

/* Line 1455 of yacc.c  */
#line 2287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setUnbound(); }
    break;

  case 332:

/* Line 1455 of yacc.c  */
#line 2288 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 333:

/* Line 1455 of yacc.c  */
#line 2289 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 334:

/* Line 1455 of yacc.c  */
#line 2290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 335:

/* Line 1455 of yacc.c  */
#line 2291 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 336:

/* Line 1455 of yacc.c  */
#line 2292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 337:

/* Line 1455 of yacc.c  */
#line 2296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 338:

/* Line 1455 of yacc.c  */
#line 2297 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setUnbound(); }
    break;

  case 339:

/* Line 1455 of yacc.c  */
#line 2298 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 340:

/* Line 1455 of yacc.c  */
#line 2299 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 341:

/* Line 1455 of yacc.c  */
#line 2300 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 342:

/* Line 1455 of yacc.c  */
#line 2301 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 343:

/* Line 1455 of yacc.c  */
#line 2302 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 344:

/* Line 1455 of yacc.c  */
#line 2307 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *val;
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            val = new Falcon::Value();
            val->setSymdef( (yyvsp[(1) - (1)].stringp) );
            // warning: the symbol is still undefined.
            COMPILER->addSymdef( val );
         }
         else {
            val = new Falcon::Value( sym );
         }
         (yyval.fal_val) = val;
     }
    break;

  case 346:

/* Line 1455 of yacc.c  */
#line 2325 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 347:

/* Line 1455 of yacc.c  */
#line 2326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtFunction *sfunc = COMPILER->getFunctionContext();
      if ( sfunc == 0 ) {
         COMPILER->raiseError(Falcon::e_fself_outside, COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value();
      }
      else
      {
         (yyval.fal_val) = new Falcon::Value( sfunc->symbol() );
      }
   }
    break;

  case 352:

/* Line 1455 of yacc.c  */
#line 2354 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( (yyvsp[(2) - (2)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 353:

/* Line 1455 of yacc.c  */
#line 2355 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, "%d", (int)(yyvsp[(2) - (2)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 354:

/* Line 1455 of yacc.c  */
#line 2356 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString("self") ); /* do not add the symbol to the compiler */ }
    break;

  case 355:

/* Line 1455 of yacc.c  */
#line 2357 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyvsp[(3) - (3)].stringp)->prepend( "." ); (yyval.fal_val)->setLBind( (yyvsp[(3) - (3)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 356:

/* Line 1455 of yacc.c  */
#line 2358 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, ".%d", (int)(yyvsp[(3) - (3)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 357:

/* Line 1455 of yacc.c  */
#line 2359 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(".self") ); /* do not add the symbol to the compiler */ }
    break;

  case 358:

/* Line 1455 of yacc.c  */
#line 2360 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 359:

/* Line 1455 of yacc.c  */
#line 2361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_fbind, new Falcon::Value((yyvsp[(1) - (3)].stringp)), (yyvsp[(3) - (3)].fal_val)) ); }
    break;

  case 360:

/* Line 1455 of yacc.c  */
#line 2362 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            // is this an immediate string sum ?
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isString() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str += *(yyvsp[(4) - (4)].fal_val)->asString();
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.writeNumber( (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
         }
    break;

  case 361:

/* Line 1455 of yacc.c  */
#line 2388 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 362:

/* Line 1455 of yacc.c  */
#line 2389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( (yyvsp[(1) - (4)].fal_val)->asString()->length() );
                  for( int i = 0; i < (yyvsp[(4) - (4)].fal_val)->asInteger(); ++i )
                  {
                     str.append( *(yyvsp[(1) - (4)].fal_val)->asString()  );
                  }
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 363:

/* Line 1455 of yacc.c  */
#line 2409 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if( (yyvsp[(1) - (4)].fal_val)->asString()->length() == 0 )
               {
                  COMPILER->raiseError( Falcon::e_invop );
               }
               else {

                  if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
                  {
                     Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                     str.setCharAt( str.length()-1, str.getCharAt(str.length()-1) + (yyvsp[(4) - (4)].fal_val)->asInteger() );
                     (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                     delete (yyvsp[(4) - (4)].fal_val);
                     (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
                  }
                  else
                     (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
               }
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 364:

/* Line 1455 of yacc.c  */
#line 2433 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.append( (Falcon::uint32) (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 365:

/* Line 1455 of yacc.c  */
#line 2450 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 366:

/* Line 1455 of yacc.c  */
#line 2451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 367:

/* Line 1455 of yacc.c  */
#line 2452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 368:

/* Line 1455 of yacc.c  */
#line 2453 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 369:

/* Line 1455 of yacc.c  */
#line 2454 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 370:

/* Line 1455 of yacc.c  */
#line 2455 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 371:

/* Line 1455 of yacc.c  */
#line 2456 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 372:

/* Line 1455 of yacc.c  */
#line 2457 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 373:

/* Line 1455 of yacc.c  */
#line 2458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 374:

/* Line 1455 of yacc.c  */
#line 2459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 375:

/* Line 1455 of yacc.c  */
#line 2460 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 376:

/* Line 1455 of yacc.c  */
#line 2461 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 377:

/* Line 1455 of yacc.c  */
#line 2462 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 378:

/* Line 1455 of yacc.c  */
#line 2463 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 379:

/* Line 1455 of yacc.c  */
#line 2464 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 380:

/* Line 1455 of yacc.c  */
#line 2465 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 381:

/* Line 1455 of yacc.c  */
#line 2466 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 382:

/* Line 1455 of yacc.c  */
#line 2467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 383:

/* Line 1455 of yacc.c  */
#line 2468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 384:

/* Line 1455 of yacc.c  */
#line 2469 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 385:

/* Line 1455 of yacc.c  */
#line 2470 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 386:

/* Line 1455 of yacc.c  */
#line 2471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 387:

/* Line 1455 of yacc.c  */
#line 2472 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); }
    break;

  case 388:

/* Line 1455 of yacc.c  */
#line 2473 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 389:

/* Line 1455 of yacc.c  */
#line 2474 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); }
    break;

  case 390:

/* Line 1455 of yacc.c  */
#line 2475 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 391:

/* Line 1455 of yacc.c  */
#line 2476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 392:

/* Line 1455 of yacc.c  */
#line 2477 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eval, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 393:

/* Line 1455 of yacc.c  */
#line 2478 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 394:

/* Line 1455 of yacc.c  */
#line 2479 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_deoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 395:

/* Line 1455 of yacc.c  */
#line 2480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_isoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 396:

/* Line 1455 of yacc.c  */
#line 2481 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_xoroob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 403:

/* Line 1455 of yacc.c  */
#line 2489 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 404:

/* Line 1455 of yacc.c  */
#line 2494 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   }
    break;

  case 405:

/* Line 1455 of yacc.c  */
#line 2498 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   }
    break;

  case 406:

/* Line 1455 of yacc.c  */
#line 2503 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 407:

/* Line 1455 of yacc.c  */
#line 2509 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 410:

/* Line 1455 of yacc.c  */
#line 2521 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   }
    break;

  case 411:

/* Line 1455 of yacc.c  */
#line 2526 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   }
    break;

  case 412:

/* Line 1455 of yacc.c  */
#line 2533 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 413:

/* Line 1455 of yacc.c  */
#line 2534 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 414:

/* Line 1455 of yacc.c  */
#line 2535 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 415:

/* Line 1455 of yacc.c  */
#line 2536 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 416:

/* Line 1455 of yacc.c  */
#line 2537 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 417:

/* Line 1455 of yacc.c  */
#line 2538 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 418:

/* Line 1455 of yacc.c  */
#line 2539 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 419:

/* Line 1455 of yacc.c  */
#line 2540 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 420:

/* Line 1455 of yacc.c  */
#line 2541 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 421:

/* Line 1455 of yacc.c  */
#line 2542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 422:

/* Line 1455 of yacc.c  */
#line 2543 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 423:

/* Line 1455 of yacc.c  */
#line 2544 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);}
    break;

  case 424:

/* Line 1455 of yacc.c  */
#line 2549 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
    break;

  case 425:

/* Line 1455 of yacc.c  */
#line 2552 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      }
    break;

  case 426:

/* Line 1455 of yacc.c  */
#line 2555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      }
    break;

  case 427:

/* Line 1455 of yacc.c  */
#line 2558 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      }
    break;

  case 428:

/* Line 1455 of yacc.c  */
#line 2561 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      }
    break;

  case 429:

/* Line 1455 of yacc.c  */
#line 2568 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      }
    break;

  case 430:

/* Line 1455 of yacc.c  */
#line 2574 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      }
    break;

  case 431:

/* Line 1455 of yacc.c  */
#line 2578 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 432:

/* Line 1455 of yacc.c  */
#line 2579 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 433:

/* Line 1455 of yacc.c  */
#line 2588 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 434:

/* Line 1455 of yacc.c  */
#line 2623 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 435:

/* Line 1455 of yacc.c  */
#line 2631 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 436:

/* Line 1455 of yacc.c  */
#line 2665 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StatementList *stmt = COMPILER->getContextSet();
         if( stmt->size() == 1 && stmt->back()->type() == Falcon::Statement::t_autoexp )
         {
            // wrap it around a return, so A is not nilled.
            Falcon::StmtAutoexpr *ae = static_cast<Falcon::StmtAutoexpr *>( stmt->pop_back() );
            stmt->push_back( new Falcon::StmtReturn( 1, ae->value()->clone() ) );

            // we don't need the expression anymore.
            delete ae;
         }

         (yyval.fal_val) = COMPILER->closeClosure();
      }
    break;

  case 438:

/* Line 1455 of yacc.c  */
#line 2684 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 439:

/* Line 1455 of yacc.c  */
#line 2688 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 441:

/* Line 1455 of yacc.c  */
#line 2696 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 442:

/* Line 1455 of yacc.c  */
#line 2700 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 443:

/* Line 1455 of yacc.c  */
#line 2707 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 444:

/* Line 1455 of yacc.c  */
#line 2741 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
    break;

  case 445:

/* Line 1455 of yacc.c  */
#line 2757 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   }
    break;

  case 446:

/* Line 1455 of yacc.c  */
#line 2762 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 447:

/* Line 1455 of yacc.c  */
#line 2769 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 448:

/* Line 1455 of yacc.c  */
#line 2776 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 449:

/* Line 1455 of yacc.c  */
#line 2785 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); }
    break;

  case 450:

/* Line 1455 of yacc.c  */
#line 2787 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 451:

/* Line 1455 of yacc.c  */
#line 2791 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 452:

/* Line 1455 of yacc.c  */
#line 2798 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); }
    break;

  case 453:

/* Line 1455 of yacc.c  */
#line 2800 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 454:

/* Line 1455 of yacc.c  */
#line 2804 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 455:

/* Line 1455 of yacc.c  */
#line 2812 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); }
    break;

  case 456:

/* Line 1455 of yacc.c  */
#line 2813 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); }
    break;

  case 457:

/* Line 1455 of yacc.c  */
#line 2815 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      }
    break;

  case 458:

/* Line 1455 of yacc.c  */
#line 2822 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 459:

/* Line 1455 of yacc.c  */
#line 2823 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 460:

/* Line 1455 of yacc.c  */
#line 2827 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 461:

/* Line 1455 of yacc.c  */
#line 2828 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 464:

/* Line 1455 of yacc.c  */
#line 2835 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      }
    break;

  case 465:

/* Line 1455 of yacc.c  */
#line 2841 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      }
    break;

  case 466:

/* Line 1455 of yacc.c  */
#line 2848 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); }
    break;

  case 467:

/* Line 1455 of yacc.c  */
#line 2849 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); }
    break;



/* Line 1455 of yacc.c  */
#line 7524 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 1675 of yacc.c  */
#line 2853 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


